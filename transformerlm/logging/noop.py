from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional

from .base import Logger


class NoOpLogger(Logger):
    def start_run(self, config: Dict[str, Any]) -> Dict[str, str]:
        # Deterministic-ish name; caller may override.
        return {"run_name": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        pass

    def finish(self) -> None:
        pass

