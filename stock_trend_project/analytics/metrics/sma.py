import numpy as np

def compute_sma(prices: np.ndarray, window: int) -> np.ndarray:
    #Simple Moving Average using a fast sliding window (O(n)).
    #Returns an array with NaN for the first window-1 elements.
    arr = np.asarray(prices, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")

    n = arr.size
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out

    wsum = arr[:window].sum()
    out[window - 1] = wsum / window

    for i in range(window, n):
        # slide: add new element, remove the one leaving the window
        wsum += arr[i] - arr[i - window]
        out[i] = wsum / window

    return out


#for validation/compare
def compute_sma_naive(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Naïve O(n*k) SMA for comparison/validation.
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
