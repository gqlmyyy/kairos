# Trading Bot V3 - core/exceptions.py

class TradingBotError(Exception):
    pass

class MT5Error(TradingBotError):
    pass

class MT5ConnectionError(MT5Error):
    pass

class OrderError(TradingBotError):
    pass

class OrderRejectedError(OrderError):
    pass

class OrderNotFilledError(OrderError):
    pass

class RiskLimitError(TradingBotError):
    pass

class DailyLossLimitError(RiskLimitError):
    pass

class DrawdownLimitError(RiskLimitError):
    pass

class DataFetchError(TradingBotError):
    pass

class ConfigurationError(TradingBotError):
    pass
