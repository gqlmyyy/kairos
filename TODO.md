- [ ] Inspect scripts/debug_exit_model_live_check.py and ensure it isolates market_regime="Unknown" from ADX failure
- [ ] Modify scripts/debug_exit_model_live_check.py to monkeypatch calculate_adx (or adapter ADX compute) to return a valid ADX (e.g., 20.0)
- [ ] Run scripts/debug_exit_model_live_check.py and capture full stdout
- [x] Verify output reason mentions market_regime/Unknown (not ADX) and features_incomplete=True with probabilities/ confidence None
- [x] Update TODO to done when verified


