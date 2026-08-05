#!/usr/bin/env python3
"""
Simulate one trading cycle to see signals
"""

from config import SYMBOLS
from analysis.technical.indicators import get_trend_score, get_momentum_score, get_volatility_score
from analysis.multi_timeframe.analyzer import get_multi_timeframe_analysis

print('='*70)
print('Trading Bot - Full Analysis Cycle')
print('='*70 + '\n')

for symbol in SYMBOLS:
    print(f'\n{symbol}:')
    print('-'*70)
    
    # Get technical scores
    trend_score, trend_dir = get_trend_score(symbol)
    mom_score, mom_dir = get_momentum_score(symbol)
    vol_score = get_volatility_score(symbol)
    
    print(f'  Technical Analysis:')
    print(f'    Trend:      {trend_score} ({trend_dir})')
    print(f'    Momentum:   {mom_score} ({mom_dir})')
    print(f'    Volatility: {vol_score}')
    
    # Get MTF decision
    mtf = get_multi_timeframe_analysis(symbol)
    print(f'  Multi-Timeframe:')
    print(f'    H4:  {mtf.h4_direction} (score={mtf.h4_score})')
    print(f'    H1:  {mtf.h1_direction} (score={mtf.h1_score})')
    print(f'    M15: {mtf.m15_direction} (score={mtf.m15_score})')
    print(f'    Aligned: {mtf.aligned} ({mtf.strength})')

print('\n' + '='*70)
