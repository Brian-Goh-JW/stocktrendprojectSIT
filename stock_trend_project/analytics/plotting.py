from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

DARK_TEMPLATE = "plotly_dark"

# Hide the Plotly modebar (camera/zoom/pan icons), keep interactions
_PLOT_CONFIG = dict(
    displayModeBar=False,   # <-- hides the toolbar
    displaylogo=False,
    responsive=True,
    scrollZoom=True,        # keep wheel zoom
    toImageButtonOptions=dict(scale=2, format="png"),
)

_SYMBOLS = {
    "USD": "$", "SGD": "S$", "EUR": "€", "GBP": "£", "JPY": "¥", "CNY": "¥",
    "AUD": "A$", "CAD": "C$"
}

def _cur_label(code: str) -> str:
    code = (code or "").upper()
    return _SYMBOLS.get(code, code or "USD")

def _legend_top_left(fig: go.Figure):
    """
    Place the legend horizontally at the TOP-LEFT, above the plotting area,
    so it never overlaps a centered title.
    """
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.0,              # left
            xanchor="left",
            y=1.12,             # a bit above the plotting area
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color="#eaf5ff"),
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        )
    )

def _range_tools(fig: go.Figure):
    # Remove rangeselector buttons; keep rangeslider + good hover
    fig.update_layout(font=dict(color="#eaf5ff"))
    fig.update_xaxes(
        rangeslider=dict(visible=True, bgcolor="rgba(255,255,255,0.06)", thickness=0.08),
        rangeselector=dict(visible=False),
        showspikes=True, spikemode="across", spikesnap="cursor", spikedash="solid", spikethickness=1
    )
    fig.update_layout(hovermode="x unified")

def fig_to_html(fig: go.Figure) -> str:
    return fig.to_html(include_plotlyjs="cdn", full_html=False, config=_PLOT_CONFIG)

def plot_price_vs_sma(
    df: pd.DataFrame,
    sma_cols=None,
    title: str = "Price & SMA",
    currency: str = "USD",
    sma_warmup_masks: list[np.ndarray] | None = None,
) -> str:
    # Plot Close price and up to two SMA lines.
    sma_cols = sma_cols or []
    sma_warmup_masks = sma_warmup_masks or [None] * len(sma_cols)

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    cur = _cur_label(currency)

    fig = go.Figure()
    # Close
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines", name="Close",
        line=dict(width=2, color="#58a6ff")
    ))

    palette = ["#f6c177", "#7ee787"]  # add more if needed

    def _label_from_col(col: str) -> str:
        return f"SMA({col.split('_')[1]})" if "_" in col and len(col.split('_')) > 1 else col

    # Add SMAs (up to two)
    for i, col in enumerate(sma_cols[:2]):
        if col not in df.columns:
            continue

        color = palette[i % len(palette)]
        label = _label_from_col(col)
        warm_mask = sma_warmup_masks[i] if (sma_warmup_masks and i < len(sma_warmup_masks)) else None

        line_style = dict(color=color, width=2, dash="dash")

        if warm_mask is None:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df[col],
                mode="lines",
                name=label,
                legendgroup=label,
                line=line_style
            ))
        else:
            y = df[col].to_numpy()
            warm_mask = np.asarray(warm_mask, dtype=bool)

            # warm-up segment (no legend entry)
            y_warm = np.where(warm_mask, y, float("nan"))
            fig.add_trace(go.Scatter(
                x=df["Date"], y=y_warm, mode="lines",
                name=label,
                legendgroup=label,
                showlegend=False,
                line=line_style,
                hoverinfo="x+y+name"
            ))

            # full-window segment (with legend)
            y_full = np.where(~warm_mask, y, float("nan"))
            fig.add_trace(go.Scatter(
                x=df["Date"], y=y_full, mode="lines",
                name=label,
                legendgroup=label,
                showlegend=True,
                line=line_style,
                hoverinfo="x+y+name"
            ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=title, x=0.5, y=0.97, xanchor="center", yanchor="top", pad=dict(b=4)),
        xaxis_title="Date",
        yaxis_title=f"Price ({cur})",
        margin=dict(l=40, r=24, t=120, b=56),  # extra top margin for legend area
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#11161c",
    )
    fig.update_layout(title_font=dict(size=18, color="#eaf5ff", family="Segoe UI"))
    fig.update_yaxes(tickprefix=_SYMBOLS.get(currency.upper(), "") or None)

    _range_tools(fig)
    _legend_top_left(fig)
    return fig_to_html(fig)

def plot_runs_overlay(
    df: pd.DataFrame,
    run_df: pd.DataFrame,
    title: str = "Candlesticks",
    min_run_len: int = 2,
    currency: str = "USD",
) -> str:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    cur = _cur_label(currency)

    fig = go.Figure(data=[
        go.Candlestick(
            x=df["Date"],
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="#2ecc71",
            decreasing_line_color="#ff6b6b",
            increasing_fillcolor="#2ecc71",
            decreasing_fillcolor="#ff6b6b",
            name="", showlegend=False
        )
    ])
    # legend proxies
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(color="#2ecc71"), name="Up candles"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(color="#ff6b6b"), name="Down candles"))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=dict(text=title, x=0.5, y=0.97, xanchor="center", yanchor="top", pad=dict(b=4)),
        xaxis_title="Date",
        yaxis_title=f"Price ({cur})",
        margin=dict(l=40, r=24, t=120, b=56),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#11161c",
    )
    fig.update_layout(title_font=dict(size=18, color="#eaf5ff", family="Segoe UI"))
    fig.update_yaxes(tickprefix=_SYMBOLS.get(currency.upper(), "") or None)

    _range_tools(fig)
    _legend_top_left(fig)
    return fig_to_html(fig)