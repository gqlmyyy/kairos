# Trading Bot V3 - data/news/scoring.py

from typing import List
from core.models import NewsItem

def score_news_item(item: NewsItem) -> float:
    """Combined relevance score with high impact boost"""
    base_score = item.source_weight * item.decay
    # أخبار عالية التأثير تأخذ وزن إضافي
    if item.is_high_impact:
        base_score *= 1.5
    return base_score

def sort_news_by_relevance(news_list: List[NewsItem]) -> List[NewsItem]:
    return sorted(news_list, key=score_news_item, reverse=True)

def filter_relevant_news(news_list: List[NewsItem], top_n: int = 20) -> List[NewsItem]:
    """فلترة وترتيب الأخبار الأكثر أهمية"""
    sorted_items = sort_news_by_relevance(news_list)
    # أبقِ الأخبار عالية التأثير دائماً
    high_impact = [n for n in sorted_items if n.is_high_impact]
    normal = [n for n in sorted_items if not n.is_high_impact]
    combined = high_impact + normal
    return combined[:top_n]
