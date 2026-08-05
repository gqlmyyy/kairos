from __future__ import annotations

from typing import Dict, Any

from data.storage.database import get_execution_dataset


class MLDatasetBuilder:
    def __init__(self) -> None:
        pass

    def on_trade_closed(self, payload: Dict[str, Any]) -> None:
        # Current architecture stores in execution_dataset via DB listener.
        # Training sample build is handled later by existing pipeline.
        return

