import numpy as np
import pandas as pd
from typing import Dict, Tuple

def compute_runs(prices: np.ndarray) -> Tuple[Dict[str, int], pd.DataFrame]:
    """
    Identify consecutive upward and downward runs.
    Returns:
      - stats: dict with counts, total days, and longest streaks
      - run_df: DataFrame with columns [start_idx, end_idx, direction, length]
    """
    prices = np.asarray(prices, dtype=float)
    change = np.diff(prices, prepend=np.nan)
    direction = np.zeros_like(prices, dtype=int)
    direction[1:] = np.where(change[1:] > 0, 1, np.where(change[1:] < 0, -1, 0))

    runs = []
    i = 1
    while i < len(direction):
        j = i
        while j + 1 < len(direction) and direction[j+1] == direction[i]:
            j += 1
        runs.append({
            "start_idx": i,
            "end_idx": j,
            "direction": int(direction[i]),
            "length": int(j - i + 1)
        })
        i = j + 1

    df = pd.DataFrame(runs)
    up_runs = df[df["direction"] == 1]
    down_runs = df[df["direction"] == -1]
    stats = {
        "total_up_runs": int(len(up_runs)),
        "total_down_runs": int(len(down_runs)),
        "days_up": int(up_runs["length"].sum()) if not up_runs.empty else 0,
        "days_down": int(down_runs["length"].sum()) if not down_runs.empty else 0,
        "longest_up_run": int(up_runs["length"].max()) if not up_runs.empty else 0,
        "longest_down_run": int(down_runs["length"].max()) if not down_runs.empty else 0,
    }
    return stats, df