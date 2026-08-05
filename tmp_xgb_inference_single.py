import sqlite3
import numpy as np
import xgboost as xgb

from config import DB_FILE
from analysis.features.ml_dataset_builder import build_ml_row
from analysis.models import xgboost_inference


def main():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM execution_dataset WHERE status='closed' AND order_id IS NOT NULL LIMIT 1")
    row = c.fetchone()
    cols = [d[0] for d in c.description]
    d = dict(zip(cols, row))
    conn.close()

    built = build_ml_row(d)
    if built is None:
        raise SystemExit("build_ml_row returned None")
    X, y = built

    model = xgboost_inference.load_model("models/xgb_model.json")
    if model is None:
        raise SystemExit("load_model returned None")

    out = xgboost_inference.predict_trade(d, model=model)

    X_np = np.asarray([X], dtype=float)
    dm = xgb.DMatrix(X_np)
    raw = model.predict(dm)

    print("execution_row expected_session:", d.get("expected_session"))
    print("execution_row actual_session:", d.get("actual_session"))
    print("execution_row expected_market_regime:", d.get("expected_market_regime"))
    print("execution_row actual_market_regime:", d.get("actual_market_regime"))

    print("X:", X)
    print("y:", y)

    print("predict_trade out:", out)
    print("direct model.predict(DMatrix(X))[0]:", float(np.asarray(raw).ravel()[0]))


if __name__ == "__main__":
    main()

