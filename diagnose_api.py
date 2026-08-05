#!/usr/bin/env python3
"""
Trading Bot V3 - API Diagnostic Tool
====================================

This script helps diagnose issues with QuantDinger API connectivity and data retrieval.

Usage:
    python diagnose_api.py
"""

import sys
import time
from datetime import datetime
from config import QUANTDINGER_URL, QUANTDINGER_USERNAME, QUANTDINGER_PASSWORD, SYMBOLS, TF_TREND, TF_DECISION, TF_TIMING
from utils.logger import get_logger
from execution.quantdinger_client import login, get_headers
from data.market.client import check_quantdinger_health, get_candles, get_indicators

logger = get_logger("diagnose")

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def diagnose():
    """Run full diagnostics"""
    print_section("Trading Bot V3 - API Diagnostic Tool")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Check configuration
    print_section("1. Configuration Check")
    print(f"QuantDinger URL: {QUANTDINGER_URL}")
    print(f"Username: {QUANTDINGER_USERNAME}")
    print(f"Trading Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframes: TREND={TF_TREND}, DECISION={TF_DECISION}, TIMING={TF_TIMING}")
    
    # 2. Test authentication
    print_section("2. Authentication Check")
    try:
        token = login()
        print(f"✓ Successfully logged in to QuantDinger")
        print(f"  Token: {token[:20]}...")
    except Exception as e:
        print(f"✗ Login failed: {e}")
        print(f"  Please verify QUANTDINGER_USERNAME and QUANTDINGER_PASSWORD in config.py")
        return False
    
    # 3. Health check
    print_section("3. QuantDinger Health Check")
    health = check_quantdinger_health()
    print(f"Status: {health['status'].upper()}")
    print(f"Connection: {'✓' if health['connection'] else '✗'}")
    print(f"Auth: {'✓' if health['auth'] else '✗'}")
    print(f"Data: {'✓' if health['data'] else '✗'}")
    if health['errors']:
        print(f"Errors:")
        for err in health['errors']:
            print(f"  - {err}")
    
    if health['status'] != 'healthy':
        print(f"\n⚠️  API is not fully healthy. This explains the 'No data found' errors.")
        return False
    
    # 4. Test data retrieval for each symbol and timeframe
    print_section("4. Data Retrieval Test")
    for symbol in SYMBOLS:
        print(f"\n{symbol}:")
        for tf in [TF_TREND, TF_DECISION, TF_TIMING]:
            try:
                candles = get_candles(symbol, tf, 10)
                if candles:
                    print(f"  {tf:5} ✓ {len(candles):3} candles")
                else:
                    print(f"  {tf:5} ✗ No data returned")
            except Exception as e:
                print(f"  {tf:5} ✗ Error: {e}")
    
    # 5. Test indicator calculation
    print_section("5. Indicator Calculation Test")
    for symbol in SYMBOLS[:1]:  # Test with first symbol only
        print(f"\n{symbol}:")
        try:
            indicators = get_indicators(symbol, TF_DECISION)
            if indicators:
                print(f"  ✓ Indicators calculated successfully:")
                for key, val in indicators.items():
                    print(f"    - {key}: {val}")
            else:
                print(f"  ✗ Failed to calculate indicators")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # 6. Summary and recommendations
    print_section("6. Diagnostics Summary")
    if health['status'] == 'healthy':
        print("✓ All systems operational!")
        print("\nThe 'No data found' errors should not occur with healthy API.")
        print("If you still see them, check:")
        print("  - QuantDinger service status")
        print("  - Market hours (data may not be available outside market hours)")
        print("  - Symbol names and market types in MARKET_MAP")
    else:
        print(f"✗ API Health Status: {health['status'].upper()}")
        print("\nRecommended actions:")
        
        if health['status'] == 'unreachable':
            print("  1. Ensure QuantDinger is running at " + QUANTDINGER_URL)
            print("  2. Check network connectivity")
            print("  3. Verify QUANTDINGER_URL in config.py")
        elif health['status'] == 'auth_failed':
            print("  1. Verify QUANTDINGER_USERNAME and QUANTDINGER_PASSWORD")
            print("  2. Check QuantDinger user credentials")
            print("  3. Ensure user has proper permissions")
        elif health['status'] == 'no_data':
            print("  1. Check if market is open")
            print("  2. Verify symbols are supported by QuantDinger")
            print("  3. Check if data is available for requested timeframes")
            print("  4. Review QuantDinger logs for errors")
        
        print("\n  5. Check logs in ./logs/ for detailed error messages")
    
    print(f"\n{'='*60}\n")
    return health['status'] == 'healthy'

if __name__ == "__main__":
    success = diagnose()
    sys.exit(0 if success else 1)
