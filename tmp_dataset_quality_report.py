from __future__ import annotations

from collections import Counter

from data.storage.database import get_conn
from data_quality import explain_rejected_row
from analysis.features.ml_dataset_builder import build_ml_row


def main():
    conn = get_conn()
    c = conn.cursor()
    rows = c.execute("SELECT * FROM execution_dataset").fetchall()
    conn.close()

    total = len(rows)
    accepted_by_quality = 0
    rejected_reasons_quality = []

    accepted_by_builder = 0
    rejected_reasons_builder = []

    for r in rows:
        row = {k: r[k] for k in r.keys()}

        acc_q, _reasons_q = explain_rejected_row(row)["accepted"], explain_rejected_row(row)["missing_or_invalid_fields"]
        if acc_q:
            accepted_by_quality += 1
        else:
            rejected_reasons_quality.append(tuple(_reasons_q))

        built = build_ml_row(row)
        if built is not None:
            accepted_by_builder += 1
        else:
            # capture only missing critical keys that are None via build_ml_row contract
            # (no explicit reasons available from ml_dataset_builder)
            # so we use data_quality's reasons as best-available explanation.
            rej = explain_rejected_row(row)["missing_or_invalid_fields"]
            rejected_reasons_builder.append(tuple(rej))

    def top(counter_list, n=10):
        ctr = Counter(counter_list)
        return ctr.most_common(n)

    print("execution_dataset total_rows:", total)
    print("accepted_by_data_quality:", accepted_by_quality)
    print("top_rejection_reasons_quality:", top(rejected_reasons_quality))

    print("accepted_by_ml_dataset_builder:", accepted_by_builder)
    print("top_rejection_reasons_builder(using data_quality reasons):", top(rejected_reasons_builder))


if __name__ == "__main__":
    main()

