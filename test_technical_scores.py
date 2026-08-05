#!/usr/bin/env python3
"""
Test technical scores with real data
"""

from analysis.technical.indicators import get_trend_score, get_momentum_score, get_volatility_score
from config import SYMBOLS

print('='*60)
print('Technical Scores (with Real Data)')
print('='*60 + '\n')

for symbol in SYMBOLS:
    trend_score, trend_dir = get_trend_score(symbol)
    mom_score = get_momentum_score(symbol)
    vol_score = get_volatility_score(symbol)
    
    print(f'{symbol}:')
    print(f'  Trend:      {trend_score} ({trend_dir})')
    print(f'  Momentum:   {mom_score}')
    print(f'  Volatility: {vol_score}')
    print()

print('='*60)
