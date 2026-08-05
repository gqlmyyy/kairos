import sqlite3
import numpy as np
import xgboost as xgb

from config import DB_FILE
from analysis.features.ml_dataset_builder import build_ml_row
from analysis.models.xgboost_inference import load_model


def main():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM execution_dataset WHERE status='closed' LIMIT 1")
    row = c.fetchone()
    cols = [d[0] for d in c.description]
    d = dict(zip(cols, row))
    conn.close()

    built = build_ml_row(d)
    if built is None:
        raise SystemExit("build_ml_row returned None for selected row")
    X, y = built

    model = load_model("models/xgb_model.json")
    if model is None:
        raise SystemExit("load_model returned None")

    X_np = np.asarray([X], dtype=float)

    pred_dmatrix = model.predict(xgb.DMatrix(X_np))
    # numpy/ndarray direct predict is expected to fail for Booster.
    # We catch to keep the test informative.
    try:
        pred_numpy = model.predict(X_np)
    except Exception as e:
        pred_numpy = None
        numpy_err = repr(e)

    print("pred_dmatrix:", pred_dmatrix)
    print("pred_numpy:", pred_numpy)

    if pred_numpy is not None:
        try:
            diff = pred_dmatrix - pred_numpy
            print("diff:", diff)
        except Exception as e:
            print("diff compute failed:", e)
    else:
        print("numpy predict error:", numpy_err)


    print("X:", X)
    print("y:", y)

    # helpful for categorical debugging
    print("db expected_session:", d.get("expected_session"))
    print("db actual_session:", d.get("actual_session"))
    print("db expected_market_regime:", d.get("expected_market_regime"))
    print("db actual_market_regime:", d.get("actual_market_regime"))


if __name__ == "__main__":
    main()

