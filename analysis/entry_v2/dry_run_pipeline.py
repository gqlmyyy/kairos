from __future__ import annotations

"""analysis/entry_v2/dry_run_pipeline.py

Dry-run pipeline for Entry v2 (NO training, NO model saving).

Flow:
- Historical Candles -> Feature Engineering -> Label Generation -> Dataset Build
- Then produce dataset audit and PASS/FAIL verdict.

Implementation note:
This repository currently only contains:
- dataset_builder.py (raw unified candle dataset pre-feature)
- feature_engineering.py (feature generation from dataset CSV)
- entry_labels.py (TP-first/SL-first simulation working on candle sequences)
- dataset_audit.py (statistical audit)

To remain strictly "NO runtime integration" and "NO training",
this script orchestrates *dataset/csv/parquet generation* only.

However, generating labels from the unified candle dataset requires selecting
entry timestamps and simulating TP/SL using OHLC only.

This script assumes a simplistic fixed TP/SL offset configuration provided
as parameters.

STOP:
- after dataset audit reports.
"""

import argparse
import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from utils.logger import get_logger

from .dataset_builder import build_dataset
from .feature_engineering import generate_features
from .entry_labels import TP_SL_Config, Holding_Config, generate_labels_for_symbol_tf, compute_label_stats
from .dataset_audit import audit_dataset

logger = get_logger("entry_v2.dry_run_pipeline")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/entry_v2_dry_run")
    p.add_argument("--months_min", type=int, default=12)
    p.add_argument("--tp_offset", type=float, default=0.0010)
    p.add_argument("--sl_offset", type=float, default=0.0010)
    p.add_argument("--max_bars", type=int, default=12)  # max holding in M15 bars for label sim
    p.add_argument("--dataset_builder_csv", default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Dataset build (raw candles)
    raw_meta = build_dataset(
        output_dir=args.out_dir,
        months_min=args.months_min,
    )
    raw_csv = raw_meta["exports"]["csv"]

    # 2) Feature engineering
    feat_dir = os.path.join(args.out_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)
    generate_features(dataset_csv_path=raw_csv, output_dir=feat_dir)

    # 3) Run label generation + dataset audit using the engineered feature dataset.
    # Feature engineering returns the engineered dataframe directly (no re-reading).
    feat_result = generate_features(dataset_csv_path=raw_csv, output_dir=feat_dir)
    df_feat = feat_result["df"]

    # 4) Label generation (TP-first/SL-first)
    # Note: entry_labels expects candle sequences; it uses TP/SL offsets and max holding.
    # We provide the same unified timestamps/symbols as engineered features.
    # The label generator returns labels aligned to df_feat.
    labels_df = generate_labels_for_symbol_tf(
        df_feat,
        tp_offset=args.tp_offset,
        sl_offset=args.sl_offset,
        max_bars=args.max_bars,
    )

    # 5) Merge features + labels inside the label module contract (if needed) and audit.
    audited = audit_dataset(labels_df)

    logger.info("Entry v2 dry-run PASS/FAIL: %s", audited.get("verdict"))
    return audited



if __name__ == "__main__":
    main()

