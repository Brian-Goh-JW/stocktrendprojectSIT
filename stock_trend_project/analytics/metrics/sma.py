from __future__ import annotations
import numpy as np

def compute_sma(
    prices: np.ndarray,
    window: int,
    min_periods: int | None = None,
    return_warmup_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    #Simple Moving Average with optional warm-up.

    # - min_periods=None (default): classic SMA => NaN for the first (window-1) points.
    # - min_periods=1: show partial averages from day 1 using however many points are available.
    # - return_warmup_mask=True: also return a boolean array 'warm' where values use <window points.

    #Complexity: O(n) via cumulative sum (no per-step loop over the window).
    arr = np.asarray(prices, dtype=float)
    n = arr.size
    out = np.full(n, np.nan, dtype=float)

    if window <= 0:
        raise ValueError("window must be positive")
    if n == 0:
        return (out, np.zeros(n, dtype=bool)) if return_warmup_mask else out

    if min_periods is None:
        min_periods = window  # classic behavior

    # cumulative sum: sum[i:j] = csum[j] - csum[i]
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    idx = np.arange(n)
    starts = np.maximum(0, idx + 1 - window)           # start index of each window
    lens = np.minimum(idx + 1, window).astype(float)   # window length at each position
    sums = csum[idx + 1] - csum[starts]
    vals = sums / lens

    valid = (idx + 1) >= min_periods
    out[valid] = vals[valid]

    if return_warmup_mask:
        warm = (lens < window) & valid  # True for early partial-window values only
        return out, warm
    return out


#naïve version for validation write-up
def compute_sma_naive(prices: np.ndarray, window: int) -> np.ndarray:
    #Naïve O(n*k) SMA for comparison/validation.
    arr = np.asarray(prices, dtype=float)
    n = arr.size
    out = np.full(n, np.nan, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")
    if n < window:
        return out
    for i in range(window - 1, n):
        out[i] = arr[i - window + 1 : i + 1].mean()
    return out