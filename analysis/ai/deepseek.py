import json
import re
import time
from typing import List, Optional

import requests

from utils.logger import get_logger
from config import DEEPSEEK_API_KEY
from core.models import NewsItem, AINewsAnalysis

logger = get_logger("deepseek")

MAX_RETRIES = 2
MAX_NEWS_ITEMS = 3
MAX_HEADLINE_LEN = 80
MAX_PROMPT_CHARS = 700


def _safe_val(x: Optional[object], as_str: bool = True) -> str:
    """Backward compatible helper."""
    if x is None:
        return "N/A"
    try:
        return str(x) if as_str else x
    except Exception:
        return "N/A"


def _is_number_or_numeric_string(v: object) -> bool:
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return float(v) == float(v)
    try:
        s = str(v).strip()
        if not s or s.upper() == "N/A":
            return False
        return float(s) == float(s)
    except Exception:
        return False


def _is_json_complete(text: str) -> bool:
    if not text:
        return False
    if not text.strip().startswith('{') or not text.strip().endswith('}'):
        return False
    open_braces = text.count('{')
    close_braces = text.count('}')
    if open_braces != close_braces:
        return False
    cleaned = re.sub(r'\\"', '', text)
    double_quotes = cleaned.count('"')
    if double_quotes % 2 != 0:
        return False
    truncated_patterns = [
        '"reason":',
        '"key_factors":',
        '"risk_factors":',
    ]
    for pattern in truncated_patterns:
        if pattern in text and text.rfind(pattern) > text.rfind('}'):
            return False
    return True


def _try_parse_json(content: str) -> Optional[dict]:
    if not content:
        return None

    content = content.strip()
    content = re.sub(r'```json?\s*', '', content)
    content = re.sub(r'```\s*', '', content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find('{')
    end = content.rfind('}')
    if start >= 0 and end > start:
        json_str = content[start : end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    if not _is_json_complete(content):
        # Best-effort repair
        fixed = content
        open_braces = content.count('{')
        close_braces = content.count('}')
        missing = open_braces - close_braces

        if missing > 0 and close_braces == 0 and '"reason"' in content:
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', content)
            if reason_match:
                reason_val = reason_match.group(1)
                fixed = re.sub(r'"reason"\s*:\s*"[^"]*"', f'"reason": "{reason_val}"', content)
                fixed += ',"key_factors":[],"risk_factors":[]}'
            else:
                fixed += '}'

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    return None


def _escape_prompt_text(text: str) -> str:
    if not text:
        return text
    text = text.replace('\\', ' ')
    text = text.replace('"', ' ').replace("'", ' ')
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
    text = re.sub(r'[,;:\[\]{}()]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _make_analysis_request(prompt: str) -> Optional[dict]:
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            # Real backoff instead of a flat 0.5s — gives the API room to
            # recover from rate limiting / load instead of hammering it.
            time.sleep(2 * attempt)

        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    # Bump max_tokens on retry in case the first failure was
                    # caused by truncation (finish_reason == "length").
                    "max_tokens": 300 if attempt == 0 else 600,
                    "temperature": 0.3,
                },
                timeout=30,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                logger.warning(
                    f"DeepSeek rate limited (429), retry_after={retry_after}, "
                    f"body={response.text[:300]}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(float(retry_after) if retry_after else 3.0)
                    continue
                return None

            if response.status_code != 200:
                logger.warning(
                    f"DeepSeek API error: status={response.status_code}, "
                    f"body={response.text[:300]}"
                )
                if attempt < MAX_RETRIES - 1:
                    continue
                return None

            data = response.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason", "")

            if not content:
                logger.warning(
                    f"DeepSeek empty response: finish_reason={finish_reason}, "
                    f"raw={response.text[:300]}"
                )
                if attempt < MAX_RETRIES - 1:
                    continue
                return None

            if finish_reason == "length":
                logger.warning(
                    f"DeepSeek truncated by max_tokens, finish_reason=length, "
                    f"content={content[:150]}"
                )
            elif finish_reason and finish_reason != "stop":
                logger.warning(
                    f"DeepSeek finish_reason={finish_reason}, content={content[:150]}"
                )

            if not _is_json_complete(content):
                logger.warning(
                    f"Retry {attempt+1}: JSON incomplete/truncated, raw={content[:300]}"
                )

            result = _try_parse_json(content)
            if result is not None:
                logger.info(f"DEEPSEEK OK: {content[:100]}...")
                return result

            # Attempt to extract JSON
            try:
                match = re.search(r'\{.*?\}', content, re.DOTALL)
                if match:
                    extracted = match.group(0)
                    result = _try_parse_json(extracted)
                    if result is not None:
                        logger.info(f"DEEPSEEK extracted JSON OK: {extracted[:100]}...")
                        return result
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"DeepSeek attempt {attempt+1} failed: {e}")

    return None


def analyze_news(news_list: List[NewsItem], symbol: str, snapshot) -> AINewsAnalysis:
    """Analyze news for a symbol using DeepSeek API."""

    try:
        # 1) filter top news
        sorted_news = sorted(
            news_list,
            key=lambda x: x.source_weight * x.decay,
            reverse=True,
        )[:MAX_NEWS_ITEMS]

        headlines_raw = [
            f"[{n.source.upper()}] {n.headline[:MAX_HEADLINE_LEN]}" for n in sorted_news
        ]
        headlines = _escape_prompt_text(" | ".join(headlines_raw))[:MAX_PROMPT_CHARS]

        # 2) snapshot guards
        if snapshot is None or not hasattr(snapshot, "data"):
            return _default_analysis("no_snapshot")

        sym_data = snapshot.data.get(symbol, {})
        if not isinstance(sym_data, dict):
            return _default_analysis("no_snapshot")

        # 3) locate H4 data for price/RSI/ATR (current market state)
        h4_data = sym_data.get("H4")
        if not isinstance(h4_data, dict):
            return _default_analysis("no_snapshot")

        def _find_first_value(container: object, substrings: List[str]) -> Optional[object]:
            """Return first value whose key contains any substring (case-insensitive).
            Recurses into nested dict/list.
            """
            if container is None:
                return None
            if isinstance(container, dict):
                for k, v in container.items():
                    try:
                        k_l = str(k).lower()
                    except Exception:
                        continue
                    for sub in substrings:
                        if sub in k_l and v is not None:
                            return v
                for v in container.values():
                    found = _find_first_value(v, substrings)
                    if found is not None:
                        return found
            elif isinstance(container, (list, tuple)):
                for it in container:
                    found = _find_first_value(it, substrings)
                    if found is not None:
                        return found
            return None

        def _as_num_str_or_na(v: object) -> str:
            return _safe_val(v, True)

        current_price = _as_num_str_or_na(
            _find_first_value(h4_data, ["price", "close", "bid"])
        )
        rsi_val = _as_num_str_or_na(_find_first_value(h4_data, ["rsi"]))
        atr_val = _as_num_str_or_na(_find_first_value(h4_data, ["atr"]))

        # 4) trend per tf: H4/H1/M15
        def _trend_for_tf(tf: str) -> str:
            data_tf = sym_data.get(tf, {})
            if not isinstance(data_tf, dict):
                data_tf = {}
            t = _find_first_value(data_tf, ["trend", "direction"])
            if t is None:
                return "neutral"
            t_str = str(t).strip()
            if not t_str or t_str.upper() == "N/A":
                return "neutral"
            return t_str

        h4_trend = _trend_for_tf("H4")
        h1_trend = _trend_for_tf("H1")
        m15_trend = _trend_for_tf("M15")
        mtf_trend = f"{h4_trend}/{h1_trend}/{m15_trend}"

        # 5) validation numeric context (strict)
        if not _is_number_or_numeric_string(current_price) or float(current_price) == 0:
            logger.warning(f"DeepSeek context invalid for {symbol}: missing/invalid price")
            return _default_analysis("context_invalid")
        if not _is_number_or_numeric_string(rsi_val) or float(rsi_val) == 0:
            logger.warning(f"DeepSeek context invalid for {symbol}: missing/invalid RSI")
            return _default_analysis("context_invalid")
        if not _is_number_or_numeric_string(atr_val) or float(atr_val) == 0:
            logger.warning(f"DeepSeek context invalid for {symbol}: missing/invalid ATR")
            return _default_analysis("context_invalid")
        if not mtf_trend or mtf_trend.upper() == "N/A":
            logger.warning(f"DeepSeek context invalid for {symbol}: missing/invalid Trend")
            return _default_analysis("context_invalid")

        # 6) daily change (%): best-effort extraction from snapshot if present
        daily_change = "N/A"
        daily_candidates = [
            ("daily_change", ["day", "daily", "change"]),
            ("change_day", ["day", "change"]),
            ("percent_change", ["percent", "pct", "change"]),
        ]
        # Try to find in any TF dict for broader coverage
        for tf_name in ["H4", "H1", "M15", "D1", "H0"]:
            tf_data = sym_data.get(tf_name)
            if not isinstance(tf_data, dict):
                continue
            # use same helper on tf_data
            for _, subs in daily_candidates:
                v = _find_first_value(tf_data, subs)
                if _is_number_or_numeric_string(v) and float(_safe_val(v, True)) != 0:
                    daily_change = _safe_val(v, True)
                    break
            if daily_change != "N/A":
                break

        prompt = (
            "You are DeepSeek, a strict trading assistant.\n"
            "Use BOTH the news and the provided market/technical context to choose direction.\n\n"
            "Market context (required):\n"
            f"- Symbol: {symbol}\n"
            f"- Current price: {current_price}\n"
            f"- Daily change (%): {daily_change}\n"
            f"- RSI: {rsi_val}\n"
            f"- ATR: {atr_val}\n"
            f"- Trend (H4/H1/M15): {h4_trend}/{h1_trend}/{m15_trend}\n\n"
            "News (top headlines):\n"
            f"{headlines}\n\n"
            "Strict instructions (MUST FOLLOW):\n"
            "1) If the news and market context suggest a clear direction, return bullish or bearish.\n"
            "2) Only return neutral if the news is truly mixed or unclear. Prefer a directional bias when possible.\n"
            "3) Return ONLY valid JSON and NOTHING else.\n"
            "4) JSON must match this schema exactly (types included):\n"
            '{"impact_score": 0-100, "bias": "bullish|bearish|neutral", "confidence": 0.0-1.0, "reason": "short Arabic text"}'
        )

        # 7) call deepseek (outer retries)
        for attempt in range(3):
            try:
                resp = _make_analysis_request(prompt)
                if resp is None:
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                    return _default_analysis("api_failed")
                result = resp
                break
            except Exception:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                return _default_analysis("api_failed")

        bias = result.get("bias", "neutral")
        if bias not in ["bullish", "bearish", "neutral"]:
            bias = "neutral"

        impact_score = float(result.get("impact_score", 0) or 0)
        impact_score = max(0.0, min(100.0, impact_score))

        confidence = float(result.get("confidence", 0) or 0)
        confidence = max(0.0, min(1.0, confidence))

        reason = str(result.get("reason", ""))[:50]

        # Backward compatible fields
        key_factors = result.get("key_factors", [])
        if not isinstance(key_factors, list):
            key_factors = []
        risk_factors = result.get("risk_factors", [])
        if not isinstance(risk_factors, list):
            risk_factors = []

        logger.info(f"AI {symbol}: bias={bias} impact={impact_score} conf={confidence:.2f}")

        return AINewsAnalysis(
            impact_score=impact_score,
            news_impact_score=impact_score,
            bias=bias,
            confidence=confidence,
            reason=reason,
            key_factors=list(key_factors)[:2],
            risk_factors=list(risk_factors)[:2],
        )

    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return _default_analysis(str(e)[:30])



def _default_analysis(error: str) -> AINewsAnalysis:
    return AINewsAnalysis(
        impact_score=0,
        news_impact_score=0,
        bias="neutral",
        confidence=0,
        reason=f"Error: {error[:30]}",
        key_factors=[],
        risk_factors=[],
    )