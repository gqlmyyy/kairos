# Trading Bot V3 - analysis/sentiment/analyzer.py
# Sentiment analysis based on keyword matching

from typing import List
from utils.logger import get_logger
from core.models import NewsItem, SentimentData

logger = get_logger("sentiment")

BULLISH_KEYWORDS = [
    "rally", "surge", "gain", "rise", "strong", "bullish",
    "optimism", "growth", "positive", "recovery", "rebound",
    "boost", "uptick", "outperform", "upgrade"
]

BEARISH_KEYWORDS = [
    "fall", "drop", "decline", "weak", "bearish", "concern",
    "fear", "risk", "uncertainty", "crash", "sell-off", "negative",
    "downgrade", "slump", "plunge", "recession"
]

SYMBOL_KEYWORDS = {
    "EURUSD": ["euro", "eur", "usd", "dollar", "ecb", "fed", "european"],
    "XAUUSD": ["gold", "xau", "safe haven", "inflation", "precious"],
    "GBPUSD": ["pound", "gbp", "uk", "boe", "bank of england", "british"],
    "USDJPY": ["yen", "jpy", "japan", "boj", "japanese"],
}

def analyze_sentiment(news_list: List[NewsItem], symbol: str) -> SentimentData:
    try:
        symbol_kws = SYMBOL_KEYWORDS.get(symbol, [])
        bullish_count = 0
        bearish_count = 0

        relevant_news = []
        for news in news_list:
            headline = news.headline.lower()
            if not symbol_kws or any(kw in headline for kw in symbol_kws):
                relevant_news.append(headline)

        if not relevant_news:
            relevant_news = [n.headline.lower() for n in news_list[:5]]

        for headline in relevant_news:
            bullish_count += sum(1 for kw in BULLISH_KEYWORDS if kw in headline)
            bearish_count += sum(1 for kw in BEARISH_KEYWORDS if kw in headline)

        total = bullish_count + bearish_count
        if total == 0:
            return SentimentData(score=50, direction="neutral")

        bullish_ratio = bullish_count / total
        if bullish_ratio > 0.6:
            score = 50 + (bullish_ratio * 50)
            direction = "bullish"
        elif bullish_ratio < 0.4:
            score = 50 + ((1 - bullish_ratio) * 50)
            direction = "bearish"
        else:
            score = 50
            direction = "neutral"

        score = round(min(100, max(0, score)))
        logger.info(f"Sentiment {symbol}: {direction} score={score} bull={bullish_count} bear={bearish_count}")
        return SentimentData(score=score, direction=direction,
                            bullish_count=bullish_count, bearish_count=bearish_count)

    except Exception as e:
        logger.error(f"Sentiment error: {e}")
        return SentimentData(score=50, direction="neutral")
