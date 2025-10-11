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
    # ensures that we are working with a numeric 1-D array of prices
    prices = np.asarray(prices, dtype=float)
    # day-to-day change; first value has no "yesterday", so it's NaN
    change = np.diff(prices, prepend=np.nan)
    # convert each day’s change into a direction code:
    # +1 if price went up vs yesterday, -1 if down, 0 if flat (no change)
    direction = np.zeros_like(prices, dtype=int)
    direction[1:] = np.where(change[1:] > 0, 1, np.where(change[1:] < 0, -1, 0))

    runs = []
    i = 1   # start at 1 because change[0] is NaN (no previous day to compare)
    while i < len(direction):
        # grow a run forward from i while the direction stays the same
        j = i
        while j + 1 < len(direction) and direction[j+1] == direction[i]:
            j += 1
        # save this run (from i to j, inclusive)
        runs.append({
            "start_idx": i,
            "end_idx": j,
            "direction": int(direction[i]),
            "length": int(j - i + 1)
        })
        # jump to the first day after this run and continue
        i = j + 1

    # make a table of all runs found
    df = pd.DataFrame(runs)
    # split into up and down runs for easy aggregation
    up_runs = df[df["direction"] == 1]
    down_runs = df[df["direction"] == -1]

    # build the summary numbers
    stats = {
        "total_up_runs": int(len(up_runs)),
        "total_down_runs": int(len(down_runs)),
        "days_up": int(up_runs["length"].sum()) if not up_runs.empty else 0,
        "days_down": int(down_runs["length"].sum()) if not down_runs.empty else 0,
        "longest_up_run": int(up_runs["length"].max()) if not up_runs.empty else 0,
        "longest_down_run": int(down_runs["length"].max()) if not down_runs.empty else 0,
    }
    return stats, df