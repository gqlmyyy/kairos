# Kairos — Remediation Baseline

State recorded **before** any remediation edit, so every later claim can be
diffed against it.

## Git

| | |
|---|---|
| Branch | `claude/drop-quantdinger` |
| Commit | `211543d3086e1949c35aa95aa142d5f5d9ddda7d` |
| Working tree | clean |

## Test suite

```
$ pytest tests/ -q
1 failed, 217 passed in 16.71s
```

| | |
|---|---|
| Passed | 217 |
| Failed | 1 (`test_predict_proba_parity.py` — intentional gate, exit-model artifact 13 features vs 12-feature schema) |
| Skipped | 0 |
| Errors | 0 |

## Verified current behaviour

### Entry model — BROKEN (C-01)

```
models/entry/entry_model.json expects : 65 features
live inference sends                  : 10 features
booster.feature_names                 : None
result                                : prediction returned, NOT rejected
```

Sensitivity sweep over the 10 supplied values (baseline
`[50.0, 0.0012, 0.0, 0.0, 50.0, 50.0, 50.0, 1.0, 1.0, 1.0]`, p=0.351562):

| changed input | p_win | responds? |
|---|---|---|
| rsi | 0.803450 | yes |
| atr | 0.332443 | yes |
| macd | 0.351562 | **no** |
| trend_strength | 0.557312 | yes |
| trend_score | 0.351562 | **no** |
| momentum_score | 0.351562 | **no** |
| volatility_score | 0.351562 | **no** |
| market_regime | 0.351562 | **no** |
| session | 0.349726 | marginal |
| **direction** | **0.351562** | **no — same p_win for BUY and SELL** |

### Risk engine — CORRECT after the 2026-08-07 fix

Incident conditions (`equity=99.40`) now reject rather than open:

| symbol | ATR | stop | size | outcome |
|---|---|---|---|---|
| XAUUSD | 47.35571 | 10.000 | 0.0 | rejected |
| EURUSD | 0.00175 | 0.00263 | 0.0 | rejected |
| GBPUSD | 0.00226 | 0.00339 | 0.0 | rejected |

At `equity=50_000` the same XAUUSD setup accepts 0.01 lots risking 0.14% —
under the 2% hard ceiling. Risk cap holds.

### MT5 execution — duplicate-order exposure (C-02)

`execution/mt5_direct.py:303-317`: the filling-mode loop `continue`s on both
`order_send` exception and `result is None`, with no reconciliation against the
broker before resending.

### Trade management — layer order correct, entry gates unwired (H-04)

Orchestrator steps 5→11 run in the documented order. Steps 3 and 4 are not
reachable from the live path:

```
check_entry_gate     : 0 call sites in main.py / execution/
can_open_new_entry   : 0 call sites
IntrabarState        : 0 call sites
can_open_new_position: only referenced from the unwired gate module
```

### MT5 session centralisation — not achieved (H-01)

Direct `mt5.*` calls bypassing `mt5_session.mt5_call()`:

| file | direct calls |
|---|---|
| `execution/reconciliation.py` | 25 |
| `execution/post_entry/action_executor.py` | 19 |
| `execution/mt5_direct.py` | 18 |
| `data/market/candle_boundary.py` | 6 |
| `main.py` | 5 |
| `risk/symbol_info.py` | 2 |
| `risk/risk_engine.py` | 2 |
| `execution/post_entry/post_entry_manager.py` | 1 |
| `execution/post_entry/trade_monitor.py` | 1 |

Independent `mt5.initialize()` / `mt5.login()` outside the session module:
`action_executor.py:343,352` · `reconciliation.py:77,105,731` ·
`candle_boundary.py:74` · `mt5_direct.py:423`

### Market data — forming candle included (H-02)

`data/market/mt5_client.py:121` uses `copy_rates_from_pos(symbol, tf, 0, count)`;
position 0 is the currently forming bar, and `get_indicators` reads `closes[-1]`.

### Test coverage gaps (H-05)

Zero tests reference: `mt5_direct`, `reconciliation`, `risk_engine`,
`voting_engine`, `signal_engine`, `xgboost_v2_inference`, `telegram`.

### Exception handling (M-04)

276 `except Exception` + 7 bare `except:` in runtime code; 41 end in
`pass`/`continue`.
