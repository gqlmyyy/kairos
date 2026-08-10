# Trading Bot V3 - train_from_historical.py
"""Deprecated entry point for entry-model training.

This script could not run. It imported
``analysis.features.historical_dataset_builder_new``, a module that does not
exist anywhere in the repository (only ``historical_dataset_builder`` does), so
every invocation died at import with ModuleNotFoundError before doing anything.

Beyond the broken import, the pipeline behind it trained on a 12-feature vector
assembled in ``analysis/features/ml_dataset_builder.py``, while live inference
sends the 10 features listed in ``analysis/models/entry_feature_spec.py``.
Repairing the import alone would have swapped a 65-vs-10 mismatch for a
12-vs-10 one — the same defect wearing different numbers.

Training now lives in ``scripts/train_entry_model.py``, which builds its vectors
through the same spec module live inference uses, so the two cannot drift apart.
This file is kept only so the documented command prints a pointer instead of a
stack trace.
"""

import sys

MESSAGE = """
train_from_historical.py has been replaced.

  Why: it imported a module that does not exist
       (analysis.features.historical_dataset_builder_new), so it could never
       run; and the pipeline behind it built a 12-feature vector while live
       inference sends 10.

  Use instead:

    1. On the Windows machine, with MT5 running and logged in:

         python scripts/fetch_training_candles.py

       Writes raw OHLC candles to data/historical/. Read-only — it opens no
       positions.

    2. Anywhere:

         python scripts/train_entry_model.py --dry-run   # validate only
         python scripts/train_entry_model.py             # train and install

       Indicators are recomputed with the live formulas, both BUY and SELL are
       labelled, validation is walk-forward, and the model is only installed
       after it passes the feature-contract and live-inference checks. The
       previous model is backed up to models_backup/ first.
"""


def main() -> int:
    print(MESSAGE)
    return 1


if __name__ == "__main__":
    sys.exit(main())
