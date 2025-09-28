from typing import List

def max_profit(prices: List[float]) -> float:
    #Greedy O(n) algorithm for max profit with unlimited transactions.
    profit = 0.0
    for i in range(1, len(prices)):
        inc = prices[i] - prices[i-1]
        if inc > 0:
            profit += inc
    return float(profit)

def extract_trades(prices: List[float]):
    #Identify the actual buy/sell trades corresponding to the max profit strategy.
    #Returns (total_profit, trades), where trades is a list of dicts:
    #{buy_idx, buy_price, sell_idx, sell_price, profit}
    n = len(prices)
    i = 0
    trades = []
    total = 0.0

    while i < n - 1:
        # find local valley
        while i < n - 1 and prices[i+1] <= prices[i]:
            i += 1
        buy_i = i

        # find local peak
        while i < n - 1 and prices[i+1] >= prices[i]:
            i += 1
        sell_i = i

        if sell_i > buy_i:
            b, s = prices[buy_i], prices[sell_i]
            total += (s - b)
            trades.append({
                "buy_idx": buy_i, "buy_price": float(b),
                "sell_idx": sell_i, "sell_price": float(s),
                "profit": float(s - b)
            })
    return float(total), trades
