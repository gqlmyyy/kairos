#!/usr/bin/env python
"""
Train XGBoost Exit Model from execution_dataset.
Reads historical trades, creates was_bad_exit label, trains XGBClassifier.
"""

import sqlite3
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def get_db_path():
    """Get database path."""
    try:
        from config import DB_FILE
        return DB_FILE
    except:
        return "trading_bot_v3.db"


def parse_indicators_json(json_str):
    """Parse indicators JSON string."""
    if not json_str or json_str in ('{}', '', None):
        return {}
    try:
        return json.loads(json_str)
    except:
        return {}


def parse_market_regime(value):
    """Parse market regime to numeric."""
    if value is None:
        return 0
    val = str(value).strip()
    regime_map = {'1': 1, 'trending': 1, '1.0': 1,
                  '2': 2, 'ranging': 2, '2.0': 2,
                  '3': 3, 'volatile': 3, '3.0': 3}
    return regime_map.get(val, 0)


def parse_session(value):
    """Parse session to numeric."""
    if value is None:
        return 0
    val = str(value).lower().strip()
    session_map = {'asian': 1, 'asia': 1,
                  'london': 2,
                  'ny': 3, 'new york': 3}
    return session_map.get(val, 0)


def load_closed_trades(db_path):
    """Load closed/completed trades from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT * FROM execution_dataset
        WHERE status = 'closed'
          AND actual_pnl IS NOT NULL
    """

    cursor = conn.execute(query)
    rows = cursor.fetchall()
    conn.close()

    trades = [dict(row) for row in rows]
    print(f"Loaded {len(trades)} closed trades from database")
    return trades


def create_label(trade):
    """
    Create was_bad_exit label:
    - 1 = bad exit (should have exited earlier or bad timing)
    - 0 = good exit

    Rules:
    1. exit_reason = 'take_profit' but actual_pnl < expected_tp * 0.5 -> bad
    2. exit_reason = 'stop_loss' and mfe > 20 -> bad (large drawdown)
    3. actual_pnl <= 0 and mfe > 10 -> bad (had profit that reversed)
    4. Otherwise -> good
    """
    actual_pnl = float(trade.get('actual_pnl') or 0)
    expected_tp = float(trade.get('expected_tp') or 0)
    exit_reason = str(trade.get('exit_reason') or '').lower()

    # Parse indicators for mfe/mae
    indicators = parse_indicators_json(trade.get('actual_indicators_json') or '{}')
    mfe = float(indicators.get('mfe') or indicators.get('MFE') or 0)
    mae = float(indicators.get('mae') or indicators.get('MAE') or 0)

    # If no mfe/mae in JSON, estimate from PnL
    if mfe == 0 and actual_pnl > 0:
        mfe = actual_pnl * 1.5
    if mae == 0 and actual_pnl < 0:
        mae = abs(actual_pnl) * 1.5

    # Rule 1: Exited at TP but got less than 50% of expected
    if 'take_profit' in exit_reason or exit_reason == 'tp':
        if expected_tp > 0 and actual_pnl < expected_tp * 0.5:
            return 1  # bad exit
        if actual_pnl <= 0:
            return 1  # bad exit - exited at TP but lost money

    # Rule 2: Stopped out with large drawdown
    if exit_reason in ('stop_loss', 'sl'):
        if mfe > 20:
            return 1  # bad exit - big drawdown before stop

    # Rule 3: Had profit that reversed to loss
    if actual_pnl <= 0 and mfe > 10:
        return 1  # bad exit - had profit that disappeared

    # Good exit
    return 0


def extract_features(trade):
    """
    Extract features for training.

    Features:
    - mfe, mae (from indicators JSON or estimated)
    - entry_atr (expected_atr or actual_atr)
    - entry_rsi (expected_rsi or actual_rsi)
    - entry_adx (from indicators or default 25)
    - market_regime (numeric: 0=none, 1=trending, 2=ranging, 3=volatile)
    - trade_duration (hours as float)
    - spread (spread_at_entry)
    - volume (default 1.0)
    - session (numeric: 0=unknown, 1=asian, 2=london, 3=ny)
    - trend_h1, trend_h4 (from expected_trend_strength)
    """
    features = {}

    actual_pnl = float(trade.get('actual_pnl') or 0)

    # MFE/MAE from indicators JSON
    indicators = parse_indicators_json(trade.get('actual_indicators_json') or '{}')
    mfe = float(indicators.get('mfe') or indicators.get('MFE') or 0)
    mae = float(indicators.get('mae') or indicators.get('MAE') or 0)

    # Estimate if not available
    if mfe == 0 and actual_pnl > 0:
        mfe = actual_pnl * 1.5
    if mae == 0 and actual_pnl < 0:
        mae = abs(actual_pnl) * 1.5

    features['mfe'] = mfe
    features['mae'] = mae

    # Entry ATR
    features['entry_atr'] = float(trade.get('expected_atr') or trade.get('actual_atr') or 0)

    # Entry RSI
    features['entry_rsi'] = float(trade.get('expected_rsi') or trade.get('actual_rsi') or 50)

    # Entry ADX (try to get from indicators, default 25)
    adx = indicators.get('adx') or indicators.get('ADX')
    features['entry_adx'] = float(adx) if adx else 25.0

    # Market regime
    regime = trade.get('expected_market_regime') or trade.get('actual_market_regime')
    features['market_regime'] = parse_market_regime(regime)

    # Trade duration (convert to hours)
    duration_str = str(trade.get('trade_duration') or '0')
    try:
        # Try parsing as minutes first
        duration_min = float(duration_str)
        features['trade_duration'] = duration_min / 60.0  # convert to hours
    except:
        features['trade_duration'] = 0.0

    # Spread
    features['spread'] = float(trade.get('spread_at_entry') or trade.get('expected_spread') or 0)

    # Volume (not available, use default)
    features['volume'] = 1.0

    # Session
    session = trade.get('expected_session') or trade.get('actual_session')
    features['session'] = parse_session(session)

    # Trend H1 and H4 (same value from trend_strength)
    trend = float(trade.get('expected_trend_strength') or trade.get('actual_trend_strength') or 0)
    features['trend_h1'] = trend
    features['trend_h4'] = trend

    return features


def prepare_dataset(trades):
    """Prepare features (X) and labels (y) from trades."""
    X_list = []
    y_list = []

    for trade in trades:
        try:
            features = extract_features(trade)
            label = create_label(trade)

            X_list.append(features)
            y_list.append(label)
        except Exception as e:
            print(f"Error processing trade {trade.get('id')}: {e}")
            continue

    if not X_list:
        return None, None

    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)

    print(f"Prepared {len(X)} samples with {len(X.columns)} features")
    return X, y


def train_xgboost(X, y):
    """Train XGBClassifier."""
    # Calculate class weights
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()

    if n_pos == 0:
        print("Warning: No bad exits found, using balanced weights")
        scale_pos_weight = 1
    else:
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    print(f"\nLabel distribution: good={n_neg}, bad={n_pos}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    # Train model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X, y)

    return model


def print_report(y_true, y_pred):
    """Print simplified training report."""
    print("\n" + "=" * 50)
    print("TRAINING REPORT")
    print("=" * 50)

    n_total = len(y_true)
    n_good = (y_true == 0).sum()
    n_bad = (y_true == 1).sum()
    n_pred_good = (y_pred == 0).sum()
    n_pred_bad = (y_pred == 1).sum()

    print(f"Total samples: {n_total}")
    print(f"  Good exits (actual): {n_good}")
    print(f"  Bad exits (actual): {n_bad}")
    print(f"\nPredictions:")
    print(f"  Predicted good: {n_pred_good}")
    print(f"  Predicted bad: {n_pred_bad}")

    # Accuracy
    accuracy = (y_true == y_pred).sum() / n_total
    print(f"\nAccuracy: {accuracy:.1%}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Pred Good  Pred Bad")
    print(f"  Actual Good    {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"  Actual Bad     {cm[1,0]:3d}       {cm[1,1]:3d}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Good', 'Bad']))

    print("=" * 50)


def main():
    print("=" * 50)
    print("Training XGBoost Exit Model")
    print("=" * 50)

    # Get database path
    db_path = get_db_path()
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return

    # Step 1: Load closed trades
    print("\n[1] Loading closed trades...")
    trades = load_closed_trades(db_path)

    if not trades:
        print("No closed trades found in database")
        return

    # Step 2: Prepare dataset
    print("\n[2] Preparing features and labels...")
    X, y = prepare_dataset(trades)

    if X is None or len(X) == 0:
        print("Failed to prepare dataset")
        return

    # Step 3: Train model
    print("\n[3] Training XGBClassifier...")
    model = train_xgboost(X, y)

    # Step 4: Save model
    print("\n[4] Saving model...")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost_exit_model.json")

    model.save_model(model_path)
    print(f"Model saved to: {model_path}")

    # Step 5: Print report
    y_pred = model.predict(X)
    print_report(y, y_pred)


if __name__ == "__main__":
    main()