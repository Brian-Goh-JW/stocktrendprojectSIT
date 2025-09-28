import numpy as np

def daily_returns(prices: np.ndarray) -> np.ndarray:
    #r[t] = (P[t] / P[t-1]) - 1
    arr = np.asarray(prices, dtype=float)
    n = arr.size

    # allocate once, default to NaN so edge/zero cases remain NaN.
    ret = np.empty_like(arr, dtype=float)
    ret[:] = np.nan
    if n < 2:
        return ret

    denom = arr[:-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        # ret[1:] = arr[1:] / arr[:-1], only where denom != 0
        np.divide(arr[1:], denom, out=ret[1:], where=(denom != 0))
        # r = ratio - 1 (in place)
        ret[1:] -= 1.0

    return ret