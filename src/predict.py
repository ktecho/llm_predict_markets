"""Prediction logic: forward forecast and backtest over historical data."""

from __future__ import annotations

import pandas as pd

from .data import INTERVAL_FREQ

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def future_timestamps(last_ts: pd.Timestamp, interval: str, n: int) -> pd.Series:
    """Generates n future timestamps at the interval's frequency.

    Note: uses a fixed calendar frequency. For 24/7 markets (crypto) it is
    exact; for assets with trading halts (nights/weekends) the timestamps
    are an approximation, a known limitation of this approach.
    """
    freq = INTERVAL_FREQ[interval]
    idx = pd.date_range(start=last_ts, periods=n + 1, freq=freq)[1:]
    return pd.Series(idx, name="timestamps")


def _slice_ohlcv(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df.iloc[start:end][OHLCV_COLUMNS].reset_index(drop=True)


def forecast(
    predictor,
    df: pd.DataFrame,
    interval: str,
    pred_len: int = 120,
    lookback: int = 400,
    verbose: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    sample_count: int = 1,
) -> pd.DataFrame:
    """Predicts the next ``pred_len`` candles from the last ``lookback`` ones.

    ``df`` must come from :func:`src.data.download_ohlcv` (``timestamps``
    + OHLCV columns). Returns the prediction DataFrame indexed by the
    future timestamps.

    ``temperature``/``top_p``/``sample_count`` only affect the Kronos backend
    (generative sampling); the other backends ignore them.
    """
    if lookback > predictor.max_context:
        raise ValueError(
            f"lookback ({lookback}) exceeds the model's maximum context "
            f"({predictor.max_context})."
        )
    if len(df) < lookback:
        raise ValueError(
            f"Not enough data: {len(df)} candles available, "
            f"{lookback} needed for the lookback."
        )

    hist = df.iloc[-lookback:].reset_index(drop=True)
    x_df = hist[OHLCV_COLUMNS]
    x_timestamp = hist["timestamps"]
    y_timestamp = future_timestamps(hist["timestamps"].iloc[-1], interval, pred_len)

    return predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        verbose=verbose,
        temperature=temperature,
        top_p=top_p,
        sample_count=sample_count,
    )


def backtest(
    predictor,
    df: pd.DataFrame,
    interval: str,
    pred_len: int = 120,
    lookback: int = 400,
    verbose: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.9,
    sample_count: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predicts a known historical window to compare against reality.

    Uses the candles ``[-lookback-pred_len : -pred_len]`` as context and
    predicts the last ``pred_len`` candles, returned together with the
    real ones for evaluation.

    Returns ``(pred_df, actual_df)`` with indexes aligned by timestamp.
    """
    if len(df) < lookback + pred_len:
        raise ValueError(
            f"Not enough data for backtest: {len(df)} candles, "
            f"lookback+pred_len = {lookback + pred_len} needed."
        )

    context_end = len(df) - pred_len
    context_start = context_end - lookback

    x_df = _slice_ohlcv(df, context_start, context_end)
    x_timestamp = df.iloc[context_start:context_end]["timestamps"].reset_index(drop=True)
    y_timestamp = df.iloc[context_end:]["timestamps"].reset_index(drop=True)

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        verbose=verbose,
        temperature=temperature,
        top_p=top_p,
        sample_count=sample_count,
    )

    actual_df = df.iloc[context_end:][["timestamps"] + OHLCV_COLUMNS].reset_index(drop=True)
    pred_df = pred_df.copy()
    pred_df.index = pd.DatetimeIndex(pred_df.index)
    actual_df.index = pd.DatetimeIndex(actual_df["timestamps"])
    return pred_df, actual_df
