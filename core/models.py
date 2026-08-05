# Trading Bot V3 - core/models.py
# Data models for the entire system

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class NewsItem:
    headline: str
    summary: str
    source: str
    source_weight: float = 0.5
    decay: float = 1.0
    is_high_impact: bool = False
    published: float = 0.0

@dataclass
class AINewsAnalysis:
    impact_score: float = 0.0       # 0-100
    bias: str = "neutral"            # bullish/bearish/neutral
    confidence: float = 0.0          # 0-1
    reason: str = ""
    key_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    news_impact_score: float = 0.0

@dataclass
class TechnicalData:
    symbol: str = ""
    timeframe: str = "H1"
    trend_score: float = 50.0        # 0-100
    trend_direction: str = "neutral"
    momentum_score: float = 50.0     # 0-100
    momentum_direction: str = "neutral"
    volatility_score: float = 50.0   # 0-100 (higher = safer)
    atr: float = 0.0
    rsi: float = 50.0
    regime: str = "UNKNOWN"

@dataclass
class MultiTimeframeData:
    h4_direction: str = "neutral"
    h4_score: float = 50.0
    h1_direction: str = "neutral"
    h1_score: float = 50.0
    m15_direction: str = "neutral"
    m15_score: float = 50.0
    aligned: bool = False            # All TFs agree
    strength: str = "weak"           # weak / moderate / strong

@dataclass
class SentimentData:
    score: float = 50.0
    direction: str = "neutral"
    bullish_count: int = 0
    bearish_count: int = 0

@dataclass
class DecisionVote:
    ai_score: float = 0.0
    trend_score: float = 0.0
    momentum_score: float = 0.0
    sentiment_score: float = 0.0
    volatility_score: float = 0.0

@dataclass
class DecisionResult:
    symbol: str = ""
    direction: str = "NEUTRAL"
    final_score: float = 0.0
    ai_score: float = 0.0
    ai_confidence: float = 0.0
    trend_score: float = 0.0
    momentum_score: float = 0.0
    sentiment_score: float = 0.0
    volatility_score: float = 0.0
    confidence: float = 0.0          # Overall confidence 0-1
    mtf_aligned: bool = False
    regime: str = "UNKNOWN"
    reason: str = ""
    weights_used: dict = field(default_factory=dict)

@dataclass
class TradeOrder:
    symbol: str = ""
    direction: str = ""              # BUY / SELL
    size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    comment: str = ""

@dataclass
class OrderResult:
    success: bool = False
    order_id: Optional[str] = None
    price: float = 0.0
    status: str = "unknown"          # sent / filled / failed / rejected
    error: str = ""

@dataclass
class AccountInfo:
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: float = 0.0
    leverage: int = 100
    currency: str = "USD"

@dataclass
class TradeRecord:
    id: int = 0
    symbol: str = ""
    direction: str = ""
    size: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    atr: float = 0.0
    final_score: float = 0.0
    ai_score: float = 0.0
    ai_confidence: float = 0.0
    reason: str = ""
    status: str = "open"
    pnl: float = 0.0
    opened_at: Optional[str] = None
    closed_at: Optional[str] = None

@dataclass
class DailyStats:
    date: str = ""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    consecutive_losses: int = 0
    max_drawdown: float = 0.0
    best_symbol: str = ""
    worst_symbol: str = ""

@dataclass
class PerformanceMetrics:
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0

@dataclass
class BotState:
    trading_paused: bool = False
    pause_until: Optional[float] = None
    cycle_count: int = 0
    last_cycle: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    mt5_connected: bool = False
    qd_connected: bool = False
