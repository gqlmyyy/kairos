#!/usr/bin/env python3
"""
Test the improved API client with fallback and caching
"""

from data.market.client import (
    get_candles, get_indicators, get_atr, get_rsi, get_macd, get_price,
    check_quantdinger_health, FALLBACK_INDICATORS, _candles_cache
)
from utils.logger import get_logger
from config import SYMBOLS

logger = get_logger("test_api")

def test_indicators():
    print("\n" + "="*60)
    print("Testing Improved API Client")
    print("="*60)
    
    # Test 1: Check health
    print("\n[Test 1] Checking QuantDinger health...")
    health = check_quantdinger_health()
    print(f"  Status: {health['status']}")
    print(f"  Connection: {'✓' if health['connection'] else '✗'}")
    print(f"  Auth: {'✓' if health['auth'] else '✗'}")
    print(f"  Data: {'✓' if health['data'] else '✗'}")
    
    # Test 2: Get indicators for each symbol
    print("\n[Test 2] Getting indicators for each symbol...")
    for symbol in SYMBOLS[:1]:  # Test first symbol
        print(f"\n  {symbol}:")
        
        indicators = get_indicators(symbol, "H4")
        print(f"    RSI: {indicators.get('rsi', 'N/A')}")
        print(f"    ATR: {indicators.get('atr', 'N/A')}")
        print(f"    MACD: {indicators.get('macd', 'N/A')}")
        print(f"    Trend: {indicators.get('ma_trend', 'N/A')}")
        print(f"    Price: {indicators.get('close', 'N/A')}")
        
        # Check if it's using fallback
        if symbol in FALLBACK_INDICATORS:
            is_fallback = (
                indicators.get('rsi') == FALLBACK_INDICATORS[symbol].get('rsi') and
                indicators.get('ma_trend') == FALLBACK_INDICATORS[symbol].get('ma_trend')
            )
            print(f"    Using fallback: {'YES ⚠️' if is_fallback else 'NO ✓'}")
    
    # Test 3: Check caching
    print("\n[Test 3] Testing cache functionality...")
    print(f"  Cache entries: {len(_candles_cache)}")
    print(f"  Cached keys: {list(_candles_cache.keys())}")
    
    # Test 4: Individual getters
    print("\n[Test 4] Testing individual getter functions...")
    symbol = SYMBOLS[0]
    print(f"  {symbol}:")
    print(f"    ATR: {get_atr(symbol)}")
    print(f"    RSI: {get_rsi(symbol)}")
    print(f"    MACD: {get_macd(symbol)}")
    print(f"    Price: {get_price(symbol)}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_indicators()
