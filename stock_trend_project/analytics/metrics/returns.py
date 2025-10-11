import numpy as np

def daily_returns(prices: np.ndarray, as_percent: bool = False) -> np.ndarray:
    """
    Compute daily returns r[t] = P[t]/P[t-1] - 1, with NaN at index 0.
    Optionally return as percentages if as_percent=True.
    """
    #converts the input prices into a NumPy array of floats
    arr = np.asarray(prices, dtype=float)
    n = arr.size

    #create an array of the same size to store results
    #fill it with NaN by default so missing/invalid values stay NaN
    ret = np.empty_like(arr, dtype=float)
    ret[:] = np.nan

    #if there is less than 2 prices, cannot calculate returns
    if n < 2:
        return ret

    #the denominator will be the previous day's price for each calculation
    denom = arr[:-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        #ratio = arr[1:] / arr[:-1] where denom != 0
        np.divide(arr[1:], denom, out=ret[1:], where=(denom != 0))
        #convert ratio into return by subtracting 1
        ret[1:] -= 1.0

    #convert to percent form
    if as_percent:
        ret[1:] *= 100.0

    return ret


def daily_returns_percent(prices: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper for percentage daily returns.
    Example: 1.23 means +1.23%.
    """
    return daily_returns(prices, as_percent=True)