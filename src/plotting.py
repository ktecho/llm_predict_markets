"""Candlestick charts with Plotly: history, prediction and backtest comparison."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_HIST = "#2196F3"
COLOR_PRED = "#F44336"
COLOR_REAL = "#4CAF50"


def _add_candlestick(fig: go.Figure, df: pd.DataFrame, name: str, row: int,
                     increasing_color: str | None = None) -> None:
    kwargs = {}
    if increasing_color is not None:
        kwargs["increasing_line_color"] = increasing_color
        kwargs["decreasing_line_color"] = increasing_color
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=name,
            **kwargs,
        ),
        row=row,
        col=1,
    )


def _add_volume(fig: go.Figure, df: pd.DataFrame, name: str, color: str, row: int) -> None:
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name=name,
            marker_color=color,
            opacity=0.6,
        ),
        row=row,
        col=1,
    )


def _to_indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the DataFrame is indexed by timestamps."""
    out = df.copy()
    if "timestamps" in out.columns:
        out.index = pd.DatetimeIndex(out["timestamps"])
    return out


def _new_figure(title: str) -> go.Figure:
    """Creates the shared two-row (price + volume) subplot figure."""
    return make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.75, 0.25], subplot_titles=(title, "Volume"),
    )


def _finalize_layout(fig: go.Figure) -> go.Figure:
    """Applies the shared layout (size, legend, margins) and returns the figure."""
    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def forecast_figure(hist_df: pd.DataFrame, pred_df: pd.DataFrame,
                    title: str = "Forecast") -> go.Figure:
    """Historical candles + predicted candles (in red)."""
    hist = _to_indexed(hist_df)
    pred = _to_indexed(pred_df)

    fig = _new_figure(title)
    _add_candlestick(fig, hist, "History", row=1)
    _add_candlestick(fig, pred, "Prediction", row=1, increasing_color=COLOR_PRED)
    _add_volume(fig, hist, "Hist. volume", COLOR_HIST, row=2)
    _add_volume(fig, pred, "Pred. volume", COLOR_PRED, row=2)

    return _finalize_layout(fig)


def backtest_figure(context_df: pd.DataFrame, actual_df: pd.DataFrame,
                    pred_df: pd.DataFrame, title: str = "Backtest") -> go.Figure:
    """Context + reality (green) vs prediction (red)."""
    context = _to_indexed(context_df)
    actual = _to_indexed(actual_df)
    pred = _to_indexed(pred_df)

    fig = _new_figure(title)
    _add_candlestick(fig, context, "Context", row=1)
    _add_candlestick(fig, actual, "Actual", row=1, increasing_color=COLOR_REAL)
    _add_candlestick(fig, pred, "Prediction", row=1, increasing_color=COLOR_PRED)
    _add_volume(fig, context, "Context volume", COLOR_HIST, row=2)
    _add_volume(fig, actual, "Actual volume", COLOR_REAL, row=2)
    _add_volume(fig, pred, "Pred. volume", COLOR_PRED, row=2)

    return _finalize_layout(fig)
