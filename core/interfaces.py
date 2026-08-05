# Trading Bot V3 - core/interfaces.py
# Abstract interfaces for dependency injection

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from .models import *


class INewsProvider(ABC):
    @abstractmethod
    def fetch_news(self) -> List[NewsItem]:
        pass

class IMarketDataProvider(ABC):
    @abstractmethod
    def get_indicators(self, symbol: str, timeframe: str) -> dict:
        pass
    
    @abstractmethod
    def get_atr(self, symbol: str, timeframe: str = "H1") -> float:
        pass
    
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        pass

class IAnalysisProvider(ABC):
    @abstractmethod
    def analyze_news(self, news: List[NewsItem], symbol: str) -> AINewsAnalysis:
        pass

class IExecutionProvider(ABC):
    @abstractmethod
    def login(self) -> Optional[str]:
        pass
    
    @abstractmethod
    def open_order(self, order: TradeOrder) -> OrderResult:
        pass
    
    @abstractmethod
    def close_order(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    def modify_order(self, order_id: str, sl: float, tp: float) -> bool:
        pass
    
    @abstractmethod
    def get_positions(self) -> List[dict]:
        pass
    
    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        pass

class IStorageProvider(ABC):
    @abstractmethod
    def save_trade(self, trade: TradeRecord) -> int:
        pass
    
    @abstractmethod
    def close_trade(self, trade_id: int, pnl: float) -> None:
        pass
    
    @abstractmethod
    def get_open_trades(self) -> List[TradeRecord]:
        pass
    
    @abstractmethod
    def get_daily_stats(self) -> DailyStats:
        pass
