# Entry v2 (independent)

This package contains an independent production-grade Entry XGBoost pipeline:
- Dataset builder (12+ months, multi-timeframe H4/H1/M15)
- Feature engineering (full schema + lags + interactions + agreement)
- TP-first / SL-first labels with deterministic fallback
- Chronological split
- Optuna training
- Calibration (Platt vs Isotonic selection)
- Threshold selection from Precision-Recall curve
- Auditing (dataset audit + feature consistency checks)
- Reports (metrics + calibration + probability distribution + feature importance)
- Runtime inference and artifact loading

Artifacts:
- `models/entry_v2/...`

Runtime switch:
- `config.ENTRY_MODEL_VERSION` (v1|v2)

