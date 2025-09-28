
# Project 1 — Stock Market Trend Analysis (Flask)

An interactive Flask app that pulls daily stock prices (live via Yahoo Finance) and computes:

Simple Moving Averages (SMA) — fast sliding-window O(n)

Daily returns — vectorized

Up/Down runs (streaks) — counts & longest streaks

Max profit (Best Time to Buy & Sell Stock II) — greedy O(n)

Two interactive charts (Plotly): Close vs SMA and Up/Down Runs (Candlesticks)

A friendly Dataset Summary (latest price, 52-week stats, win rate, streaks, etc.)

## HOW TO USE (for Prof / Others)

FOR WINDOWS:
```bash #use these codes in the order
cd stock_trend_project
python -m venv .venv
# if you see an execution policy error when activating:
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate
python -m pip install -r requirements.txt
python app.py
```

FOR MACOS/LINUX:
```bash #use these codes in the order
cd stock_trend_project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

## USAGE

Enter a Ticker (e.g., AAPL).

Pick Start date and End date (calendar picker).

Set SMA #1 (days). Optionally enable SMA #2.

Set Min run length (days).

Tip: Use 2–4 to hide one-day blips and show meaningful streaks.

Click Run Analysis.

Outputs

Close vs SMA — price line with one or two SMA overlays (legend outside, zoom/pan, scroll wheel to zoom).

Up/Down Runs (Candlesticks) — green/red candles, clean run highlighting.

Dataset Summary — latest price, cumulative return, % up days, CAGR, 52-week high/low, average volume, longest streaks, simple trading model profit.

Preview — last rows of data (most recent first, scrollable).

## Validations (5 test cases)

Run:
```bash
python tests/run_validation.py
```

Included checks:
1. **SMA** vs `pandas.Series.rolling(window).mean()`
2. **Daily Returns** vs manual formula
3. **SMA** when series shorter than window → all NaN
4. **Max Profit (II)** matches known example (7.0)
5. **Trade Extraction** profit equals greedy result and yields expected number of trades

## File Structure (if needed to sort, idk)

```
stock_trend_project/
├── app.py
├── analytics/
│   ├── loader.py          # pulls data via yfinance, normalizes columns
│   ├── plotting.py        # Plotly figures (dark theme, legends outside)
│   ├── utils.py           # helpers: normalize columns, parse dates, etc.
│   └── metrics/           # core computations (each file, one responsibility)
│       ├── sma.py         # sliding-window SMA (O(n))
│       ├── returns.py     # daily returns (vectorized)
│       ├── runs.py        # up/down streaks (from sign of price diffs)
│       ├── maxprofit.py   # greedy Best Time to Buy & Sell II + trade extraction
│       └── __init__.py    # optional re-exports
├── templates/
│   └── index.html         # single-page UI
├── tests/
│   └── run_validation.py  # ≥5 correctness checks / corner cases
├── requirements.txt
└── README.md
```

## Notes

NIL for now.
```

