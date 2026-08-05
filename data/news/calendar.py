# Trading Bot V3 - data/news/calendar.py
# Economic calendar for high-impact event detection

import requests
from datetime import datetime, timedelta
from utils.logger import get_logger
from config import FINNHUB_API_KEY

logger = get_logger("calendar")

def get_economic_calendar() -> list:
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        r = requests.get(
            "https://finnhub.io/api/v1/calendar/economic",
            params={"from": today, "to": tomorrow, "token": FINNHUB_API_KEY},
            timeout=10
        )
        return r.json().get("economicCalendar", [])
    except Exception as e:
        logger.error(f"Calendar error: {e}")
        return []

def is_high_impact_soon(minutes: int = 30) -> bool:
    events = get_economic_calendar()
    now = datetime.now()
    for event in events:
        if event.get("impact") != "high":
            continue
        try:
            event_time = datetime.fromisoformat(event.get("time", ""))
            diff = (event_time - now).total_seconds() / 60
            if 0 <= diff <= minutes:
                logger.warning(f"High impact event in {diff:.0f}min: {event.get('event')}")
                return True
        except:
            continue
    return False

