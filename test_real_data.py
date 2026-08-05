#!/usr/bin/env python3
"""
Test real data from hybrid client
"""

from data.market.hybrid_client import get_indicators_hybrid
from config import SYMBOLS

print('='*60)
print('Real Data from execution_dataset')
print('='*60 + '\n')

for symbol in SYMBOLS:
    indicators = get_indicators_hybrid(symbol)
    print(f'{symbol}:')
    print(f'  RSI:   {indicators["rsi"]}')
    print(f'  MACD:  {indicators["macd"]}')
    print(f'  ATR:   {indicators["atr"]}')
    print(f'  Price: {indicators["close"]}')
    print(f'  Trend: {indicators["ma_trend"]}')
    print()

print('='*60)
print('✓ All data retrieved successfully!')
print('='*60)
