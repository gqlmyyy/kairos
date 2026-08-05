# TODO: Refactor market_regime_detector.py (QuantDinger-only)

- [ ] Confirm current implementation and identify all MT5 dependencies.
- [ ] Inspect how QuantDinger data is fetched inside the project (existing helpers).
- [ ] Replace MT5-based ATR/ADX computation with QuantDinger-only implementation.
- [ ] Remove MT5 imports and MT5-specific helpers from market_regime_detector.py.
- [ ] Ensure `detect_market_regime(symbol: str, atr: float = None) -> str` public API remains unchanged.
- [ ] Ensure `get_regime_settings()` remains unchanged.
- [ ] Guarantee `detect_market_regime` never returns "Unknown"; on failures return safe default "Normal" (per requirement) and log error.
- [ ] Write the full updated file content.
- [ ] Run any available unit tests / quick import checks.
- [ ] Summarize changes in a short report.
