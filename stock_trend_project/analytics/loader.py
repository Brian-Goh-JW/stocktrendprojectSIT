import pandas as pd
try:
    import yfinance as yf
except ImportError:
    yf = None

def load_from_yfinance(ticker: str, period: str = "3y") -> pd.DataFrame:
    #Fetch adjusted OHLCV from Yahoo Finance for a SINGLE ticker.
    #Always returns standard single-level columns: Date, Open, High, Low, Close, Volume.
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    df = yf.download(
        ticker,
        period=period,
        group_by="column",
        auto_adjust=True,
        progress=False,
    )

    # If a MultiIndex slipped through, select the ticker level
    if isinstance(df.columns, pd.MultiIndex):
        # Try common layouts
        if ticker in df.columns.get_level_values(0):
            df = df[ticker]
        elif ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, axis=1, level=1)
        # After selection, columns should be like ['Open','High','Low','Close','Volume']

    df = df.reset_index()   # ensures Date column exists
    return df