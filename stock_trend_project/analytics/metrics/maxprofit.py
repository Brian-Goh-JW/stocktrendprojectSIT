from typing import List

def max_profit(prices: List[float]) -> float:
    #greedy O(n) algorithm for max profit where multiple trans in allowed
    profit = 0.0
    #loop from second day (index 1)
    for i in range(1, len(prices)):
        #calculate the price difference between tdy and ytd
        inc = prices[i] - prices[i-1]
        #if today price > yesterday, add difference to total profit
        if inc > 0:
            profit += inc
    #return the total profit from all positive price differences
    return float(profit)

def extract_trades(prices: List[float]):
    #identify the exact buy/sell trades points for each profitable trade based on greedy
    #returns list of dictionaries
    #dict contains: buy/sell index, buy/sell price, and profit
    n = len(prices)      #total no. of price points
    i = 0
    trades = []          #list to store all buy/sell trade details

    total = 0.0          #accumulator for total profit

    while i < n - 1:
        # find local valley (meaning lowest point before price rises)
        while i < n - 1 and prices[i+1] <= prices[i]:
            i += 1
        buy_i = i 

        # find local peak (meaning highest point before price drops)
        while i < n - 1 and prices[i+1] >= prices[i]:
            i += 1
        sell_i = i

        #only record trade if there is a valid rise (peak after valley)
        if sell_i > buy_i:
            b, s = prices[buy_i], prices[sell_i]
            total += (s - b)
            trades.append({
                "buy_idx": buy_i, "buy_price": float(b),
                "sell_idx": sell_i, "sell_price": float(s),
                "profit": float(s - b)
            })
    return float(total), trades