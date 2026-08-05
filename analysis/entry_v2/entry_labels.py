from __future__ import annotations

"""analysis/entry_v2/entry_labels.py

Production-quality Entry v2 label generator.

Rules (strict):
- NO training, NO Optuna, NO model creation.
- Labels are generated only from the engineered dataset features.
- NO usage of actual_pnl, floating pnl, future close comparisons, or profit sign.

Labeling logic (TP-first / SL-first):
For each observation (row) representing a potential entry:
- Read the entry price from feature column ``entry_price``.
- Read ATR at entry from feature column ``entry_atr``.
- Compute:
    TP = entry_price + (1.5 * entry_atr)
    SL = entry_price - (1.0 * entry_atr)
  (SELL reversal: if a direction exists, invert automatically.)

Direction handling:
- If a feature column ``direction`` exists and is categorical:
    - direction > 0 => BUY (use TP above, SL below)
    - direction <= 0 => SELL (reverse TP/SL)
- If no direction exists, default to BUY.

Future simulation:
- Walk forward candle-by-candle using historical OHLC.
- The first touched level determines the label:
    - if TP touched first => label=1
    - if SL touched first => label=0

Holding horizon:
- Maximum holding period: 24 H4 candles.
- If neither TP nor SL is reached within horizon, deterministic fallback:
    label depends on which level is closer to the final close in the holding window.

Output columns appended:
- label (0/1)
- label_reason ("tp_first" | "sl_first" | "fallback_neither" | "timeout")
- holding_bars (int)
- tp_hit (bool)
- sl_hit (bool)
- exit_timestamp (unix seconds float)

Artifacts saved:
- data/entry_v2/labeled_dataset.parquet
- data/entry_v2/labeled_dataset.csv

At the end returns a Label Generation Report dict.
"""

import os
import math
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
import numpy as np  # type: ignore

from utils.logger import get_logger

logger = get_logger("entry_v2.entry_labels")


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

DEFAULT_MAX_H4_CANDLES = 24
DEFAULT_TP_MULT = 1.5
DEFAULT_SL_MULT = 1.0


@dataclass(frozen=True)
class LabelGenConfig:
    max_h4_candles: int = DEFAULT_MAX_H4_CANDLES
    tp_mult: float = DEFAULT_TP_MULT
    sl_mult: float = DEFAULT_SL_MULT


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in engineered dataset: {missing}")


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _validate_basic_outputs(out: pd.DataFrame) -> None:
    if out.empty:
        raise RuntimeError("Label generation produced an empty dataset")

    if "label" not in out.columns:
        raise RuntimeError("Labeled dataset missing 'label'")

    # duplicates: full row duplication
    if out.duplicated().any():
        raise RuntimeError("Labeled dataset contains duplicate rows")

    # duplicates on (symbol,t) if present
    if "symbol" in out.columns and "t" in out.columns:
        if out.duplicated(subset=["symbol", "t"]).any():
            raise RuntimeError("Labeled dataset contains duplicate (symbol,t)")

    # label set
    lbl = out["label"].astype(float).to_numpy()
    if not np.isin(lbl, np.array([0.0, 1.0])).all():
        bad = lbl[~np.isin(lbl, np.array([0.0, 1.0]))]
        raise RuntimeError(f"Label column contains values other than {0,1}. Bad sample count={len(bad)}")

    # holding_bars constraint
    if "holding_bars" in out.columns:
        hp = out["holding_bars"].astype(int).to_numpy()
        if (hp < 0).any():
            raise RuntimeError("holding_bars contains negative values")


def _direction_from_row(row: pd.Series) -> int:
    # SELL reversal if direction exists
    if "direction" not in row.index:
        return 1  # default BUY
    try:
        return 1 if float(row["direction"]) > 0 else 0
    except Exception:
        return 1


# --------------------------------------------------------------------------------------
# Candle loading (for future simulation)
# --------------------------------------------------------------------------------------


def _load_h4_candles_for_symbol(symbol: str) -> pd.DataFrame:
    """Load H4 OHLC used for TP/SL walk-forward.

    This implementation expects an existing candle cache produced by the entry_v2 pipeline.

    Required columns:
    - t, open, high, low, close

    If your repository uses a different storage location, adjust ONLY this loader.
    """
    # Prefer unified dataset exported by dataset_builder (entry_v2_dataset_*.csv)
    # In this repo, the unified CSV does NOT have a 'timeframe' column.
    # Instead it contains h4_* columns.
    try:
        import glob

        csv_candidates = [
            "data/entry_v2/entry_v2_dataset_*.csv",
        ]

        matches: List[str] = []
        for pattern in csv_candidates:
            matches.extend(glob.glob(pattern))

        if matches:
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            latest = matches[0]
            df_all = pd.read_csv(latest)

            if "symbol" not in df_all.columns or "t" not in df_all.columns:
                raise RuntimeError("Unified dataset missing symbol/t columns")

            # take rows where h4 exists for the symbol
            df_sym = df_all[df_all["symbol"] == symbol].copy()

            # If there is a coverage flag, filter on it
            if "has_h4" in df_sym.columns:
                df_sym = df_sym[df_sym["has_h4"].astype(float) > 0.0].copy()

            required_h4 = {"h4_open", "h4_high", "h4_low", "h4_close"}
            if required_h4.issubset(set(df_sym.columns)) and not df_sym.empty:
                # keep expected columns and normalize names to candle_loader format
                df_h4 = pd.DataFrame(
                    {
                        "time": df_sym["t"].astype(float),
                        "open": df_sym["h4_open"].astype(float),
                        "high": df_sym["h4_high"].astype(float),
                        "low": df_sym["h4_low"].astype(float),
                        "close": df_sym["h4_close"].astype(float),
                        "tick_volume": df_sym["h4_volume"].astype(float) if "h4_volume" in df_sym.columns else 0.0,
                    }
                )

                # normalize to the columns used by _simulate_one
                df_h4 = df_h4.rename(columns={"time": "t"})

                # drop warm-up where OHLC might be NaN
                df_h4 = df_h4.dropna(subset=["t", "open", "high", "low", "close"])
                if not df_h4.empty:
                    return df_h4[["t", "open", "high", "low", "close", "tick_volume"]].rename(
                        columns={"tick_volume": "volume"}
                    )
    except Exception:
        # continue to candidates below
        pass


    # Try common locations inside entry_v2 artifacts
    candidates = [
        f"data/entry_v2_candles_{symbol}_H4.csv",
        f"data/entry_v2/{symbol}_H4.csv",
        f"data/entry_v2/h4_{symbol}.csv",
    ]

    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df

    # As a fallback, we use dataset_builder export if it exists.
    # Look for a unified candles file.
    unified = "data/entry_v2/historical_candles_unified.csv"
    if os.path.exists(unified):
        df_all = pd.read_csv(unified)
        df_all = df_all[(df_all["symbol"] == symbol) & (df_all["timeframe"] == "H4")]
        if df_all.empty:
            raise RuntimeError(f"No H4 candles found in unified candle file for symbol={symbol}")
        return df_all

    raise RuntimeError(
        "Cannot locate H4 candle series for walk-forward simulation. "
        "Add/adjust a candle loader in _load_h4_candles_for_symbol()."
    )


def _simulate_one(
    *,
    row: pd.Series,
    candles_h4: pd.DataFrame,
    cfg: LabelGenConfig,
) -> Dict[str, Any]:
    """Simulate TP/SL with a candle-by-candle walk-forward (first touch wins)."""

    # Identify entry candle in candles_h4 by t
    entry_t = _safe_float(row["t"])
    if math.isnan(entry_t):
        raise RuntimeError("Row has invalid 't'")

    entry_price = _safe_float(row["entry_price"])
    entry_atr = _safe_float(row["entry_atr"])
    if math.isnan(entry_price) or math.isnan(entry_atr):
        raise RuntimeError("Row has invalid entry_price or entry_atr")

    # ensure candles sorted
    candles_h4 = candles_h4.sort_values("t").reset_index(drop=True)

    # find index of candle with exact t match; if none, find first candle with t>=entry_t
    # (deterministic)
    t_arr = candles_h4["t"].astype(float).to_numpy()
    idx = int(np.searchsorted(t_arr, entry_t, side="left"))
    if idx >= len(candles_h4):
        # timeout beyond available history
        # deterministic fallback: label based on closer level to last close
        last = candles_h4.iloc[-1]
        close_end = _safe_float(last["close"])
        tp = entry_price + cfg.tp_mult * entry_atr
        sl = entry_price - cfg.sl_mult * entry_atr
        winner = abs(close_end - tp) < abs(close_end - sl)
        return {
            "label": 1.0 if winner else 0.0,
            "label_reason": "timeout",
            "holding_bars": int(len(candles_h4) - 1),
            "tp_hit": False,
            "sl_hit": False,
            "exit_timestamp": float(last["t"]),
        }

    # direction
    direction_buy = _direction_from_row(row)  # 1 means BUY

    if direction_buy == 1:
        tp = entry_price + cfg.tp_mult * entry_atr
        sl = entry_price - cfg.sl_mult * entry_atr
        tp_first_reason = "tp_first"
        sl_first_reason = "sl_first"
    else:
        # SELL: reverse
        tp = entry_price - cfg.tp_mult * entry_atr
        sl = entry_price + cfg.sl_mult * entry_atr
        tp_first_reason = "tp_first"
        sl_first_reason = "sl_first"

    max_end = min(len(candles_h4) - 1, idx + cfg.max_h4_candles)

    tp_hit = False
    sl_hit = False
    tp_ts: Optional[float] = None
    sl_ts: Optional[float] = None

    # walk forward candle-by-candle, starting with next candle after entry
    for j in range(idx + 1, max_end + 1):
        high = _safe_float(candles_h4.iloc[j]["high"])
        low = _safe_float(candles_h4.iloc[j]["low"])
        if math.isnan(high) or math.isnan(low):
            continue

        # Touch checks depend on direction.
        if direction_buy == 1:
            tp_in = high >= tp
            sl_in = low <= sl
        else:
            tp_in = low <= tp
            sl_in = high >= sl

        if tp_in and sl_in:
            # deterministic: TP first
            tp_hit = True
            tp_ts = float(candles_h4.iloc[j]["t"])
            holding = int(j - idx)
            return {
                "label": 1.0,
                "label_reason": tp_first_reason + "_same_bar",
                "holding_bars": holding,
                "tp_hit": True,
                "sl_hit": False,
                "exit_timestamp": tp_ts,
            }

        if tp_in:
            tp_hit = True
            tp_ts = float(candles_h4.iloc[j]["t"])
            holding = int(j - idx)
            return {
                "label": 1.0,
                "label_reason": tp_first_reason,
                "holding_bars": holding,
                "tp_hit": True,
                "sl_hit": False,
                "exit_timestamp": tp_ts,
            }

        if sl_in:
            sl_hit = True
            sl_ts = float(candles_h4.iloc[j]["t"])
            holding = int(j - idx)
            return {
                "label": 0.0,
                "label_reason": sl_first_reason,
                "holding_bars": holding,
                "tp_hit": False,
                "sl_hit": True,
                "exit_timestamp": sl_ts,
            }

    # fallback: neither hit
    c_end = candles_h4.iloc[max_end]
    close_end = _safe_float(c_end["close"])

    if math.isnan(close_end):
        close_end = entry_price

    # deterministic rule: closer to final close
    winner = abs(close_end - tp) < abs(close_end - sl)
    return {
        "label": 1.0 if winner else 0.0,
        "label_reason": "fallback_neither",
        "holding_bars": int(max_end - idx),
        "tp_hit": tp_hit,
        "sl_hit": sl_hit,
        "exit_timestamp": float(c_end["t"]),
    }


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def generate_entry_labels_v2(
    *,
    engineered_parquet_path: str = "data/entry_v2/features_dataset.parquet",
    engineered_csv_path: str = "data/entry_v2/features_dataset.csv",
    output_parquet_path: str = "data/entry_v2/labeled_dataset.parquet",
    output_csv_path: str = "data/entry_v2/labeled_dataset.csv",
    cfg: LabelGenConfig = LabelGenConfig(),
    parquet_preferred: bool = True,
) -> Dict[str, Any]:
    """Generate labels, validate, save, and return a label generation report."""

    if parquet_preferred and os.path.exists(engineered_parquet_path):
        df = pd.read_parquet(engineered_parquet_path)
    else:
        df = pd.read_csv(engineered_csv_path)

    _require_columns(df, ["t", "symbol", "entry_price", "entry_atr"])

    # Load candle series per symbol once
    symbols = sorted(df["symbol"].astype(str).unique().tolist())
    candles_by_symbol: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        candles_by_symbol[sym] = _load_h4_candles_for_symbol(sym)

    # Simulate per row
    reports: List[Dict[str, Any]] = []
    out_cols = [
        "label",
        "label_reason",
        "holding_bars",
        "tp_hit",
        "sl_hit",
        "exit_timestamp",
    ]

    label_arr: List[float] = []
    reason_arr: List[str] = []
    holding_arr: List[int] = []
    tp_hit_arr: List[bool] = []
    sl_hit_arr: List[bool] = []
    exit_ts_arr: List[float] = []

    for idx, row in df.iterrows():
        sym = str(row["symbol"])
        candles = candles_by_symbol[sym]
        sim = _simulate_one(row=row, candles_h4=candles, cfg=cfg)

        label_arr.append(float(sim["label"]))
        reason_arr.append(str(sim["label_reason"]))
        holding_arr.append(int(sim["holding_bars"]))
        tp_hit_arr.append(bool(sim["tp_hit"]))
        sl_hit_arr.append(bool(sim["sl_hit"]))
        exit_ts_arr.append(float(sim["exit_timestamp"]))

    df_out = df.copy()
    df_out["label"] = np.asarray(label_arr, dtype=float)
    df_out["label_reason"] = np.asarray(reason_arr, dtype=object)
    df_out["holding_bars"] = np.asarray(holding_arr, dtype=int)
    df_out["tp_hit"] = np.asarray(tp_hit_arr, dtype=bool)
    df_out["sl_hit"] = np.asarray(sl_hit_arr, dtype=bool)
    df_out["exit_timestamp"] = np.asarray(exit_ts_arr, dtype=float)

    # Validation
    _validate_basic_outputs(df_out)

    max_h = int(cfg.max_h4_candles)
    if (df_out["holding_bars"].astype(int) > max_h).any():
        raise RuntimeError("holding_bars exceeds max horizon")

    # Statistics report
    total = int(len(df_out))
    winners = int((df_out["label"].astype(float) >= 0.5).sum())
    losers = total - winners
    win_rate = winners / max(total, 1)

    avg_holding = float(df_out["holding_bars"].astype(float).mean())
    median_holding = float(df_out["holding_bars"].astype(float).median())

    tp_first_count = int((df_out["label_reason"].astype(str).str.startswith("tp_first")).sum())
    sl_first_count = int((df_out["label_reason"].astype(str).str.startswith("sl_first")).sum())
    timeout_count = int((df_out["label_reason"].astype(str) == "timeout").sum())

    report: Dict[str, Any] = {
        "total_samples": total,
        "winner_count": winners,
        "loser_count": losers,
        "win_rate": win_rate,
        "avg_holding_bars": avg_holding,
        "median_holding_bars": median_holding,
        "tp_first_count": tp_first_count,
        "sl_first_count": sl_first_count,
        "timeout_count": timeout_count,
        "label_value_set": sorted(set(df_out["label"].astype(float).unique().tolist())),
        "tp_sl_config": {
            "tp_mult": cfg.tp_mult,
            "sl_mult": cfg.sl_mult,
            "max_h4_candles": cfg.max_h4_candles,
        },
        "output_parquet_path": output_parquet_path,
        "output_csv_path": output_csv_path,
    }

    print("[Entry v2 Label Generation]",
          "Total:", total,
          "Winners:", winners,
          "Losers:", losers,
          "Win rate:", win_rate,
          "Avg holding:", avg_holding,
          "Median holding:", median_holding,
          "TP-first:", tp_first_count,
          "SL-first:", sl_first_count,
          "Timeout:", timeout_count)

    # Save
    os.makedirs(os.path.dirname(output_parquet_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

    df_out.to_parquet(output_parquet_path, index=False)
    df_out.to_csv(output_csv_path, index=False)

    return report


if __name__ == "__main__":
    # Only run label generation when executed directly.
    generate_entry_labels_v2()

