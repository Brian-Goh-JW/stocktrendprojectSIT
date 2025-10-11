from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

def required_buffer(window: int) -> int:
    """
    How many extra prior points should be fetched to get a valid SMA value
    on the very first *visible* day.
    Example: window=20 -> need 19 prior points.
    """
    # check that window size is valid
    if window <= 0:
        raise ValueError("window must be positive")
    return max(0, window - 1)

def compute_sma(
    prices: np.ndarray,
    window: int,
    min_periods: Optional[int] = None,
    return_warmup_mask: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Simple Moving Average with optional warm-up.

    - min_periods=None (default): classic SMA => NaN for the first (window-1) points.
    - min_periods=1: show partial averages from day 1 using however many points exist.
    - return_warmup_mask=True: also return a boolean array 'warm' where values used <window points.

    Complexity: O(n) via cumulative sum (no per-step loop over the window).
    """
    # inputs are converted to float NumPy array
    arr = np.asarray(prices, dtype=float)
    n = arr.size
    # prepare an array of NaNs to store SMA results
    out = np.full(n, np.nan, dtype=float)

    # handle invalid or empty inputs
    if window <= 0:
        raise ValueError("window must be positive")
    if n == 0:
        # if no prices, will return an array of NaNs
        return (out, np.zeros(n, dtype=bool)) if return_warmup_mask else out

    # standard SMA (full window)
    if min_periods is None:
        min_periods = window  # classic behavior

    # cumulative sum: sum[i:j] = csum[j] - csum[i]
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    idx = np.arange(n)
    starts = np.maximum(0, idx + 1 - window)           # start index of each window
    lens = np.minimum(idx + 1, window).astype(float)   # window length at each position
    sums = csum[idx + 1] - csum[starts]
    vals = sums / lens

    # for keeping values only with enougb data points
    valid = (idx + 1) >= min_periods
    out[valid] = vals[valid]

    # for SMA points before the window that user inputted
    if return_warmup_mask:
        warm = (lens < window) & valid  # True for early partial-window values only
        return out, warm
    return out

# --- Helper for “fetch buffer then show only user range” ---------------------

def compute_sma_for_display(
    full_prices_with_buffer: np.ndarray,
    window: int,
    display_start_idx: int,
    display_end_idx: Optional[int] = None,
    *,
    min_periods: Optional[int] = None,
    return_warmup_mask: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Compute SMA on the FULL (buffered) series, but return ONLY the slice
    the user asked to see (display range).

    Typical usage:
        # user picked window=20 and visible range [start, end]
        # loader already fetched (window-1) extra points BEFORE start
        buf = required_buffer(window)          # 19
        # your display_start_idx in the full array is 'buf'
        sma_vis = compute_sma_for_display(prices_full, window, display_start_idx=buf)

    Parameters
    ----------
    full_prices_with_buffer : np.ndarray
        Prices including the (window-1) prior points BEFORE the visible start.
    window : int
        SMA window.
    display_start_idx : int
        Index in the full array where the user's visible range begins.
        (If you fetched exactly (window-1) buffer points, this is window-1.)
    display_end_idx : Optional[int]
        End index (exclusive) of the visible range in the full array.
        If None, returns until the end.
    min_periods : Optional[int]
        See compute_sma.
    return_warmup_mask : bool
        If True, also returns a mask for warm-up points within the *visible* slice.

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        SMA values sliced to [display_start_idx:display_end_idx].
        If return_warmup_mask=True, returns (sma_slice, warmup_mask_slice).
    """
    sma_full, warm_full = (
        compute_sma(full_prices_with_buffer, window, min_periods, True)
        if return_warmup_mask
        else (compute_sma(full_prices_with_buffer, window, min_periods, False), None)
    )

    sma_slice = sma_full[display_start_idx:display_end_idx]
    if return_warmup_mask:
        warm_slice = warm_full[display_start_idx:display_end_idx]  # type: ignore[index]
        return sma_slice, warm_slice
    return sma_slice


# Naïve version for validation
def compute_sma_naive(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Naïve O(n*k) SMA for comparison/validation.
    The loop every window type, which is simpler BUT slower
    To note: only used for testing
    """
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