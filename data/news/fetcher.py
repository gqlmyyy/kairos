# Trading Bot V3 - data/news/fetcher.py
# Fetches news from RSS feeds + Finnhub

import feedparser
import requests
import time
import calendar
from datetime import datetime
from typing import List, Optional
from utils.logger import get_logger
from config import FINNHUB_API_KEY, NEWS_SOURCE_WEIGHTS, NEWS_DECAY_HOURS
from core.models import NewsItem

logger = get_logger("news_fetcher")

RSS_FEEDS = [
   # مصادر عالية الموثوقية
   ("https://feeds.reuters.com/reuters/businessNews", "reuters"),
   ("https://feeds.reuters.com/reuters/UKdomesticNews", "reuters"),
   ("https://www.forexlive.com/feed/news", "forexlive"),
   ("https://www.dailyfx.com/feeds/all", "dailyfx"),
   ("https://www.fxstreet.com/rss/news", "fxstreet"),
   ("https://www.kitco.com/rss/kitco-news.xml", "kitco"),
   ("https://feeds.bbci.co.uk/news/business/rss.xml", "bbc"),
   ("https://www.investing.com/rss/news_14.rss", "investing"),
   # مصادر جديدة
   ("https://www.marketwatch.com/rss/topstories", "marketwatch"),
   ("https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines", "marketwatch"),
   ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "cnbc"),
   ("https://www.cnbc.com/id/10000664/device/rss/rss.html", "cnbc"),
   ("https://rss.app/feeds/forex.xml", "forexfactory"),
   ("https://www.fxempire.com/api/v1/en/articles/rss", "fxempire"),
   ("https://www.goldprice.org/feeds/gold-price-news.xml", "goldprice"),
   ("https://finance.yahoo.com/news/rssindex", "yahoo"),
]

FOREX_KEYWORDS = [
   "dollar", "euro", "eur", "usd", "gbp", "pound", "yen", "jpy",
   "fed", "ecb", "boj", "interest rate", "inflation", "gdp",
   "gold", "xau", "forex", "currency", "central bank", "monetary",
   "employment", "payroll", "cpi", "fomc", "treasury", "yield",
   "trade war", "tariff", "sanctions", "opec", "oil", "risk",
   "bank", "debt", "deficit", "surplus", "pmi", "retail sales",
   "hawkish", "dovish", "rate", "basis points", "bps",
   "dxy", "dollar index", "safe haven", "risk off", "risk on",
   "geopolit", "war", "peace", "conflict", "agreement", "deal"
]

HIGH_IMPACT_KEYWORDS = [
   "fed", "fomc", "powell", "rate hike", "rate cut", "nfp",
   "non-farm", "cpi", "inflation", "gdp", "ecb", "lagarde",
   "boj", "recession", "emergency", "crisis", "hawkish", "dovish",
   "rate decision", "basis points", "quantitative", "pivot",
   "default", "collapse", "crash", "surge", "plunge", "soar",
   "war", "sanctions", "tariff", "trade deal", "opec"
]

USD_GENERAL_KEYWORDS = [
    "fed", "fomc", "powell", "federal reserve", "rate hike", "rate cut",
    "interest rate", "inflation", "cpi", "nfp", "non-farm", "payroll",
    "dollar", "usd", "dxy", "treasury", "yield", "recession",
    "trump", "tariff", "trade war", "sanctions"
]

SYMBOL_KEYWORDS = {
   "EURUSD": ["euro", "eur", "ecb", "lagarde", "europe", "eurozone", "german", "france"] + USD_GENERAL_KEYWORDS,
   "GBPUSD": ["pound", "gbp", "boe", "bailey", "britain", "uk", "england", "brexit"] + USD_GENERAL_KEYWORDS,
   "XAUUSD": ["gold", "xau", "bullion", "safe haven", "precious metal", "inflation hedge", "geopolit", "war", "conflict"] + USD_GENERAL_KEYWORDS,
   "USDJPY": ["yen", "jpy", "boj", "japan", "japanese", "ueda", "tokyo"] + USD_GENERAL_KEYWORDS,
}

def get_source_weight(source_name: str) -> float:
   for key, weight in NEWS_SOURCE_WEIGHTS.items():
       if key in source_name.lower():
           return weight
   return NEWS_SOURCE_WEIGHTS["default"]

def calculate_decay(published_time) -> float:
   try:
       if isinstance(published_time, str):
           pub = datetime.fromisoformat(published_time)
       else:
           pub = datetime.fromtimestamp(published_time)
       hours_ago = (datetime.now() - pub).total_seconds() / 3600
       return max(0, 1 - (hours_ago / NEWS_DECAY_HOURS))
   except:
       return 0.5

# News cache - avoid fetching on every cycle
_news_cache: Optional[List[NewsItem]] = None
_news_cache_time: float = 0.0
NEWS_CACHE_TTL_SEC = 600  # 10 minutes


def fetch_rss_news() -> List[NewsItem]:
    global _news_cache, _news_cache_time

    # Check cache validity
    now = time.time()
    if _news_cache is not None and (now - _news_cache_time) < NEWS_CACHE_TTL_SEC:
        return _news_cache

    # Build fresh news list
    all_news = []
    seen_headlines = set()

    for feed_url, source_name in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                headline = entry.get("title", "").strip()
                if not headline or headline in seen_headlines:
                    continue
                seen_headlines.add(headline)

                summary = entry.get("summary", "")[:300]
                combined = (headline + " " + summary).lower()

                # --- IMPORTANT ---
                # The system must keep ONLY forex/macro-relevant news.
                # However, some feeds may put content in other fields (e.g., 'description').
                # Keep this logic but also extend combined text when summary is empty.
                if not summary:
                    summary = entry.get("description", "")[:300]
                    combined = (headline + " " + summary).lower()

                # Check keywords
                matches = sum(
                    1 for kw in FOREX_KEYWORDS
                    if kw in combined
                )
                if matches < 2:
                    continue

                # Get timestamp
                pub_time = entry.get("published_parsed")
                pub_timestamp = calendar.timegm(pub_time) if pub_time else time.time()

                # Skip old news
                hours_old = (time.time() - pub_timestamp) / 3600
                if hours_old > 8:
                    continue

                decay = calculate_decay(pub_timestamp)
                if decay < 0.1:
                    continue

                all_news.append(NewsItem(
                    headline=headline,
                    summary=summary,
                    source=source_name,
                    source_weight=get_source_weight(source_name),
                    decay=decay,
                    is_high_impact=any(kw in combined for kw in HIGH_IMPACT_KEYWORDS),
                    published=pub_timestamp
                ))

        except Exception as e:
            logger.warning(f"RSS {source_name} failed: {e}")

    # إضافة أخبار Finnhub
    finnhub_news = fetch_finnhub_news()
    all_news.extend(finnhub_news)

    # ترتيب بالأحدث والأهم
    all_news.sort(key=lambda x: x.source_weight * x.decay, reverse=True)

    # Save to cache
    _news_cache = all_news
    _news_cache_time = time.time()

    logger.info(f"Fetched {len(all_news)} relevant news items")
    return all_news

def fetch_finnhub_news() -> List[NewsItem]:
   try:
       r = requests.get(
           "https://finnhub.io/api/v1/news",
           params={"category": "forex", "token": FINNHUB_API_KEY},
           timeout=10
       )
       items = r.json()
       result = []
       for item in items[:30]:
           headline = item.get("headline", "").strip()
           if not headline:
               continue
           combined = (headline + " " + item.get("summary", "")).lower()
           if not any(kw in combined for kw in FOREX_KEYWORDS):
               continue

           pub_time = item.get("datetime", time.time())
           hours_old = (time.time() - pub_time) / 3600
           if hours_old > 8:
               continue

           result.append(NewsItem(
               headline=headline,
               summary=item.get("summary", "")[:300],
               source="finnhub",
               source_weight=0.6,
               decay=calculate_decay(pub_time),
               is_high_impact=any(kw in combined for kw in HIGH_IMPACT_KEYWORDS),
               published=pub_time
           ))
       return result
   except Exception as e:
       logger.error(f"Finnhub error: {e}")
       return []

def filter_news_for_symbol(news: List[NewsItem], symbol: str) -> List[NewsItem]:
   """فلترة الأخبار المتعلقة بزوج معين"""
   keywords = SYMBOL_KEYWORDS.get(symbol, [])
   if not keywords:
       return []

   relevant: List[NewsItem] = []
   for item in news:
       combined = (item.headline + " " + item.summary).lower()
       if any(kw in combined for kw in keywords):
           relevant.append(item)

   # لو ما فيه أخبار خاصة بالزوج → تجاهل الزوج (لا تُرسل أخبار عامة)
   return relevant