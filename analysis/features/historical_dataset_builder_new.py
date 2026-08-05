# Trading Bot V3 - analysis/features/historical_dataset_builder.py
# Convert historical candles into training dataset

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import json

from utils.logger import get_logger
from analysis.technical.indicators import (
    calculate_rsi, calculate_macd, calculate_atr,
    calculate_moving_average, get_trend_score
)
from data.storage.database import get_conn

logger = get_logger("historical_dataset_builder")


class HistoricalDatasetBuilder:
    """Build ML-ready training dataset from historical candles"""
    
    def __init__(self):
        self.min_candles = 100  # Need enough for indicators
    
    def _calculate_indicators(
        self,
        candles: List[Dict[str, Any]],
        index: int
    ) -> Dict[str, float]:
        """Calculate technical indicators at a point in history"""
        
        if index < self.min_candles:
            return {}
        
        # Get lookback window
        window = candles[max(0, index - self.min_candles):index + 1]
        closes = [float(c.get("close", 0)) for c in window]
        highs = [float(c.get("high", 0)) for c in window]
        lows = [float(c.get("low", 0)) for c in window]
        
        if len(closes) < 20:
            return {}
        
        try:
            # RSI (14)
            rsi = calculate_rsi(closes, period=14) or 50.0
            
            # MACD
            macd_val, signal, histogram = calculate_macd(closes) or (0.0, 0.0, 0.0)
            
            # ATR (14)
            atr = calculate_atr(highs, lows, closes, period=14) or 0.0
            
            # Moving averages
            ma20 = calculate_moving_average(closes, 20) or closes[-1]
            ma50 = calculate_moving_average(closes, 50) or closes[-1]
            ma200 = calculate_moving_average(closes, 200) or closes[-1]
            
            # Trend determination
            current_close = closes[-1]
            trend = "up" if current_close > ma20 else "down" if current_close < ma20 else "sideways"
            
            # Volatility
            recent_closes = closes[-20:]
            volatility = (max(recent_closes) - min(recent_closes)) / (sum(recent_closes) / len(recent_closes)) if recent_closes else 0.0
            
            return {
                "rsi": float(rsi),
                "macd": float(macd_val),
                "atr": float(atr),
                "ma20": float(ma20),
                "ma50": float(ma50),
                "ma200": float(ma200),
                "trend": trend,
                "volatility": float(volatility),
                "close": float(current_close)
            }
        except Exception as e:
            logger.warning(f"Indicator calculation error: {e}")
            return {}
    
    def build_training_pairs(
        self,
        candles: List[Dict[str, Any]],
        symbol: str,
        lookback_bars: int = 20,
        lookahead_bars: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Build training samples from historical candles
        
        Each sample contains:
        - Features at entry point (RSI, MACD, ATR, etc.)
        - Label: whether price went up or down
        
        Args:
            candles: List of OHLCV candles sorted by time
            symbol: Trading symbol
            lookback_bars: Bars used for feature calculation
            lookahead_bars: Bars used for outcome determination
            
        Returns:
            List of training samples
        """
        
        if len(candles) < lookback_bars + lookahead_bars:
            logger.warning(f"Not enough candles for {symbol}: {len(candles)}")
            return []
        
        training_samples = []
        
        for i in range(lookback_bars, len(candles) - lookahead_bars):
            try:
                # Entry point (features at i)
                entry_indicators = self._calculate_indicators(candles, i)
                
                if not entry_indicators:
                    continue
                
                entry_price = float(candles[i].get("close", 0))
                entry_time = candles[i].get("time", "")
                
                # Outcome (price movement over next lookahead_bars)
                future_candles = candles[i+1:i+1+lookahead_bars]
                future_high = max([float(c.get("high", 0)) for c in future_candles])
                future_low = min([float(c.get("low", 0)) for c in future_candles])
                future_close = float(future_candles[-1].get("close", 0)) if future_candles else entry_price
                
                # Determine if trade was profitable (up)
                # Simple rule: if price closed higher than entry, it's a win
                price_move = future_close - entry_price
                win_move = future_high - entry_price
                loss_move = entry_price - future_low
                
                # Label: 1 if potential profit, 0 if potential loss
                label = 1 if price_move > 0 else 0
                
                # Profit potential
                profit_pips = win_move / 0.0001 if symbol in ["EURUSD", "GBPUSD"] else win_move
                loss_pips = loss_move / 0.0001 if symbol in ["EURUSD", "GBPUSD"] else loss_move
                
                sample = {
                    "symbol": symbol,
                    "time": entry_time,
                    "entry_price": entry_price,
                    "exit_price": future_close,
                    "label": label,
                    "profit_pips": profit_pips,
                    "loss_pips": loss_pips,
                    # Features
                    "rsi": entry_indicators.get("rsi", 50),
                    "macd": entry_indicators.get("macd", 0),
                    "atr": entry_indicators.get("atr", 0),
                    "ma20": entry_indicators.get("ma20", entry_price),
                    "ma50": entry_indicators.get("ma50", entry_price),
                    "ma200": entry_indicators.get("ma200", entry_price),
                    "volatility": entry_indicators.get("volatility", 0),
                    "trend": entry_indicators.get("trend", "sideways"),
                }
                
                training_samples.append(sample)
                
            except Exception as e:
                logger.debug(f"Sample building error at index {i}: {e}")
                continue
        
        logger.info(f"Built {len(training_samples)} training samples for {symbol}")
        return training_samples
    
    def load_csv_and_build(
        self,
        csv_path: str,
        symbol: str
    ) -> List[Dict[str, Any]]:
        """Load candles from CSV and build training samples"""
        
        if not os.path.exists(csv_path):
            logger.error(f"CSV not found: {csv_path}")
            return []
        
        candles = []
        
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    candle = {
                        "time": row.get("time"),
                        "open": float(row.get("open", 0)),
                        "high": float(row.get("high", 0)),
                        "low": float(row.get("low", 0)),
                        "close": float(row.get("close", 0)),
                        "volume": float(row.get("volume", 0))
                    }
                    candles.append(candle)
            
            logger.info(f"Loaded {len(candles)} candles from {csv_path}")
            
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return []
        
        return self.build_training_pairs(candles, symbol)
    
    def save_training_samples(
        self,
        samples: List[Dict[str, Any]],
        output_path: str
    ) -> None:
        """Save training samples to JSON"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)
        
        logger.info(f"✓ Saved {len(samples)} samples to {output_path}")
    
    def import_to_database(
        self,
        samples: List[Dict[str, Any]],
        table_name: str = "execution_dataset"
    ) -> int:
        """
        Import training samples into database
        
        Returns:
            Number of rows imported
        """
        
        conn = get_conn()
        c = conn.cursor()
        
        imported = 0
        
        for i, sample in enumerate(samples):
            try:
                order_id = f"HIST_{sample['symbol']}_{i}"
                
                c.execute("""
                    INSERT OR REPLACE INTO execution_dataset (
                        order_id, symbol, dataset_created_at,
                        expected_entry, expected_rsi, expected_macd, expected_atr,
                        expected_volatility_score, actual_exit, actual_pnl,
                        actual_ai_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    order_id,
                    sample["symbol"],
                    datetime.now().isoformat(),
                    sample["entry_price"],
                    sample["rsi"],
                    sample["macd"],
                    sample["atr"],
                    sample["volatility"],
                    sample["exit_price"],
                    sample["profit_pips"] if sample["label"] == 1 else -sample["loss_pips"],
                    float(sample["label"])
                ))
                
                imported += 1
                
            except Exception as e:
                logger.debug(f"Import error for sample {i}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"✓ Imported {imported} samples to database")
        return imported


def build_training_dataset_from_directory(
    historical_data_dir: str = "data/historical",
    output_dir: str = "data/training"
) -> None:
    """Build complete training dataset from all CSV files in directory"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    builder = HistoricalDatasetBuilder()
    
    all_samples = []
    
    for filename in os.listdir(historical_data_dir):
        if filename.endswith("_historical.csv"):
            symbol = filename.replace("_historical.csv", "")
            csv_path = os.path.join(historical_data_dir, filename)
            
            logger.info(f"\nProcessing {symbol}...")
            
            samples = builder.load_csv_and_build(csv_path, symbol)
            all_samples.extend(samples)
            
            # Save per-symbol
            symbol_output = os.path.join(output_dir, f"{symbol}_training.json")
            builder.save_training_samples(samples, symbol_output)
    
    # Save combined
    combined_output = os.path.join(output_dir, "combined_training.json")
    builder.save_training_samples(all_samples, combined_output)
    
    # Import to database
    builder.import_to_database(all_samples)
    
    logger.info(f"\n✓ Built training dataset with {len(all_samples)} total samples")


if __name__ == "__main__":
    build_training_dataset_from_directory()
