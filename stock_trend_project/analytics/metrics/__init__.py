from __future__ import annotations

from .sma import compute_sma
from .returns import daily_returns
from .maxprofit import max_profit, extract_trades
from .runs import compute_runs

__all__ = [
    "compute_sma",
    "daily_returns",
    "max_profit",
    "extract_trades",
    "compute_runs",
]