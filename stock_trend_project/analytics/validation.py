import numpy as np
import pandas as pd
from analytics.metrics import compute_sma, daily_returns, max_profit, extract_trades

def validate_tests():
    results = []

    # Test 1: SMA vs pandas.rolling().mean()
    prices = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
    window = 3
    ours = compute_sma(prices, window)
    pandas_ref = pd.Series(prices).rolling(window).mean().to_numpy()
    results.append(("SMA vs pandas rolling mean", np.allclose(ours[~np.isnan(ours)], pandas_ref[~np.isnan(pandas_ref)])))

    # Test 2: Daily returns vs manual formula
    dr = daily_returns(prices)
    manual = np.full_like(prices, np.nan)
    manual[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    results.append(("Daily returns formula check", np.allclose(dr[1:], manual[1:])))

    # Test 3: SMA when series shorter than window -> all NaN
    short = np.array([10,11], dtype=float)
    ours_short = compute_sma(short, 5)
    results.append(("SMA with short series", np.isnan(ours_short).all()))

    # Test 4: Max profit simple pattern
    p = [7,1,5,3,6,4]  # LeetCode classic (max profit II = (5-1)+(6-3)=7)
    profit = max_profit(p)
    results.append(("Max profit greedy correctness", abs(profit - 7.0) < 1e-9))

    # Test 5: Trades extraction consistency with max_profit
    total, trades = extract_trades(p)
    results.append(("Trades extraction profit equals greedy", abs(total - profit) < 1e-9 and len(trades) == 2))

    return results

if __name__ == "__main__":
    for name, ok in validate_tests():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")