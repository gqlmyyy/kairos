# Trading Bot V3 - data/market/historical_fetcher.py
# Fetch real historical data from MT5 and alternative sources for training

from __future__ import annotations

import requests
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

from utils.logger import get_logger
from config import QUANTDINGER_URL, SYMBOLS

logger = get_logger("historical_fetcher")


class MT5HistoricalFetcher:
    """Fetch historical candles from MT5 via QuantDinger"""
    
    def __init__(self):
        self.base_url = QUANTDINGER_URL
        self.session = requests.Session()
        
    def _get_headers(self) -> dict:
        from execution.quantdinger_client import get_headers
        return get_headers()
    
    def fetch_candles(
        self, 
        symbol: str,
        timeframe: str = "H1",
        days_back: int = 365,
        limit: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical candles from MT5 via QuantDinger
        
        Args:
            symbol: Trading symbol (e.g. "EURUSD")
            timeframe: M1, M5, M15, H1, H4, D1
            days_back: How many days of history to fetch
            limit: Max candles to fetch per request
            
        Returns:
            List of candle dictionaries with OHLCV data
        """
        try:
            market_map = {
                "EURUSD": "Forex", "GBPUSD": "Forex", "USDJPY": "Forex",
                "XAUUSD": "Forex", "USDCAD": "Forex", "AUDUSD": "Forex"
            }
            market = market_map.get(symbol, "Forex")
            
            params = {
                "symbol": symbol,
                "market": market,
                "timeframe": timeframe,
                "limit": min(limit, 5000)  # MT5 max is typically 5000
            }
            
            logger.info(f"Fetching {symbol} {timeframe} candles ({days_back} days)...")
            
            r = self.session.get(
                f"{self.base_url}/api/indicator/kline",
                params=params,
                headers=self._get_headers(),
                timeout=15
            )
            
            data = r.json()
            
            if data.get("code") not in [1, 200]:
                logger.error(f"MT5 fetch error: {data.get('msg')}")
                return []
            
            candles = data.get("data", [])
            logger.info(f"✓ Fetched {len(candles)} candles for {symbol} {timeframe}")
            
            return candles
            
        except Exception as e:
            logger.error(f"MT5 historical fetch error: {e}")
            return []
    
    def fetch_daily_candles(
        self,
        symbol: str,
        days_back: int = 730  # 2 years
    ) -> List[Dict[str, Any]]:
        """Fetch daily candles for longer-term training data"""
        return self.fetch_candles(symbol, "D1", days_back, limit=5000)
    
    def fetch_hourly_candles(
        self,
        symbol: str,
        days_back: int = 365
    ) -> List[Dict[str, Any]]:
        """Fetch hourly candles for medium-term training data"""
        return self.fetch_candles(symbol, "H1", days_back, limit=5000)


class AlternativeDataFetchers:
    """Fetch historical data from alternative free/paid APIs"""
    
    @staticmethod
    def fetch_from_alpha_vantage(
        symbol: str,
        interval: str = "60min",
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch data from Alpha Vantage API (free tier available)
        
        Args:
            symbol: Stock/Forex symbol
            interval: 1min, 5min, 15min, 30min, 60min, daily
            api_key: Alpha Vantage API key
            
        Returns:
            List of OHLCV candles
        """
        if not api_key:
            logger.warning("Alpha Vantage API key not provided")
            return []
        
        try:
            # For Forex
            if symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
                endpoint = "FX_INTRADAY"
                from_currency = symbol[:3]
                to_currency = symbol[3:]
                params = {
                    "function": endpoint,
                    "from_symbol": from_currency,
                    "to_symbol": to_currency,
                    "interval": interval,
                    "apikey": api_key
                }
            else:
                # For stocks
                endpoint = "TIME_SERIES_INTRADAY"
                params = {
                    "function": endpoint,
                    "symbol": symbol,
                    "interval": interval,
                    "apikey": api_key
                }
            
            r = requests.get(
                "https://www.alphavantage.co/query",
                params=params,
                timeout=10
            )
            
            data = r.json()
            
            if "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return []
            
            candles = []
            time_series_key = [k for k in data.keys() if "Time Series" in k]
            
            if time_series_key:
                ts = data[time_series_key[0]]
                for timestamp, ohlcv in ts.items():
                    candles.append({
                        "time": timestamp,
                        "open": float(ohlcv.get("1. open", 0)),
                        "high": float(ohlcv.get("2. high", 0)),
                        "low": float(ohlcv.get("3. low", 0)),
                        "close": float(ohlcv.get("4. close", 0)),
                        "volume": float(ohlcv.get("5. volume", 0))
                    })
            
            logger.info(f"✓ Fetched {len(candles)} candles from Alpha Vantage for {symbol}")
            return candles
            
        except Exception as e:
            logger.error(f"Alpha Vantage fetch error: {e}")
            return []
    
    @staticmethod
    def fetch_from_yfinance(symbol: str, period: str = "2y") -> List[Dict[str, Any]]:
        """
        Fetch historical data using yfinance (Yahoo Finance)
        
        Args:
            symbol: Stock symbol
            period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            
        Returns:
            List of OHLCV candles
        """
        try:
            import yfinance as yf
            
            # Map forex to stock equivalents
            if symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
                symbol = symbol.replace("USD", "=X")  # EURUSD -> EUR=X
            
            logger.info(f"Fetching {symbol} from yfinance (period: {period})...")
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            candles = []
            for date, row in hist.iterrows():
                candles.append({
                    "time": date.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"])
                })
            
            logger.info(f"✓ Fetched {len(candles)} candles from yfinance for {symbol}")
            return candles
            
        except ImportError:
            logger.warning("yfinance not installed. Install with: pip install yfinance")
            return []
        except Exception as e:
            logger.error(f"yfinance fetch error: {e}")
            return []
    
    @staticmethod
    def fetch_from_polygon_io(
        symbol: str,
        api_key: Optional[str] = None,
        days_back: int = 365
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical data from Polygon.io API
        
        Args:
            symbol: Stock symbol
            api_key: Polygon.io API key
            days_back: Days of history
            
        Returns:
            List of OHLCV candles
        """
        if not api_key:
            logger.warning("Polygon.io API key not provided")
            return []
        
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            
            params = {
                "from": from_date,
                "to": to_date,
                "timespan": "day",
                "limit": 50000,
                "apiKey": api_key
            }
            
            r = requests.get(
                f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}",
                params=params,
                timeout=10
            )
            
            data = r.json()
            
            if not data.get("results"):
                logger.warning(f"No data from Polygon.io for {symbol}")
                return []
            
            candles = []
            for bar in data["results"]:
                candles.append({
                    "time": datetime.fromtimestamp(bar["t"] / 1000).isoformat(),
                    "open": float(bar.get("o", 0)),
                    "high": float(bar.get("h", 0)),
                    "low": float(bar.get("l", 0)),
                    "close": float(bar.get("c", 0)),
                    "volume": float(bar.get("v", 0))
                })
            
            logger.info(f"✓ Fetched {len(candles)} candles from Polygon.io for {symbol}")
            return candles
            
        except Exception as e:
            logger.error(f"Polygon.io fetch error: {e}")
            return []


class HistoricalDataManager:
    """Manage collection and organization of historical data"""
    
    def __init__(self):
        self.mt5_fetcher = MT5HistoricalFetcher()
        self.alternative_fetchers = AlternativeDataFetchers()
    
    def fetch_all_sources(
        self,
        symbols: List[str],
        timeframes: List[str] = ["D1", "H1", "H4"],
        days_back: int = 365
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Fetch data from all available sources
        
        Returns:
            {symbol: {timeframe: [candles]}}
        """
        all_data = {}
        
        for symbol in symbols:
            logger.info(f"\n📊 Fetching historical data for {symbol}...")
            all_data[symbol] = {}
            
            # Try MT5 first
            for tf in timeframes:
                logger.info(f"  → {tf} timeframe...")
                candles = self.mt5_fetcher.fetch_candles(
                    symbol,
                    timeframe=tf,
                    days_back=days_back
                )
                all_data[symbol][tf] = candles
                time.sleep(1)  # Rate limiting
            
        return all_data
    
    def fetch_and_save_csv(
        self,
        symbols: List[str] = None,
        output_dir: str = "data/historical"
    ) -> Dict[str, str]:
        """
        Fetch historical data and save to CSV files for later use
        
        Returns:
            {symbol: filepath}
        """
        if symbols is None:
            symbols = SYMBOLS
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        for symbol in symbols:
            try:
                logger.info(f"Downloading {symbol}...")
                
                # Fetch from MT5
                candles = self.mt5_fetcher.fetch_daily_candles(symbol, days_back=365)
                candles.extend(self.mt5_fetcher.fetch_hourly_candles(symbol, days_back=30))
                
                if not candles:
                    logger.warning(f"No data fetched for {symbol}")
                    continue
                
                # Save to CSV
                import csv
                filepath = os.path.join(output_dir, f"{symbol}_historical.csv")
                
                with open(filepath, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["time", "open", "high", "low", "close", "volume"]
                    )
                    writer.writeheader()
                    
                    for candle in candles:
                        writer.writerow({
                            "time": candle.get("time"),
                            "open": candle.get("open"),
                            "high": candle.get("high"),
                            "low": candle.get("low"),
                            "close": candle.get("close"),
                            "volume": candle.get("volume")
                        })
                
                saved_files[symbol] = filepath
                logger.info(f"✓ Saved {len(candles)} candles to {filepath}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        return saved_files


def main():
    """Example usage: fetch historical data"""
    manager = HistoricalDataManager()
    
    # Fetch data for all symbols
    data = manager.fetch_all_sources(
        symbols=SYMBOLS,
        timeframes=["D1", "H1"],
        days_back=365
    )
    
    # Save to CSV
    files = manager.fetch_and_save_csv(symbols=SYMBOLS)
    
    logger.info(f"\n✓ Successfully saved {len(files)} files for training")


if __name__ == "__main__":
    main()
