"""OHLCV series download from Yahoo Finance, normalized to the app schema.

The app uses lowercase ``open, high, low, close`` columns plus ``volume`` and
``amount``, together with a Series of timestamps, fed to the forecasting models.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

# Supported intervals (yfinance) -> equivalent pandas frequency
INTERVAL_FREQ: dict[str, str] = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "1h",
    "90m": "90min",
    "1h": "1h",
    "1d": "1D",
    "5d": "5D",
    "1wk": "7D",
}

# Periods offered by yfinance, ordered from shortest to longest duration
VALID_PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]

# Known Yahoo limitations: intraday data has limited history
MAX_PERIOD_BY_INTERVAL = {
    "1m": "5d",
    "2m": "1mo",
    "5m": "1mo",
    "15m": "1mo",
    "30m": "1mo",
    "60m": "2y",
    "90m": "1mo",
    "1h": "2y",
}

REQUIRED_COLUMNS = ["open", "high", "low", "close"]


def download_ohlcv(ticker: str, interval: str = "1h", period: str = "1mo") -> pd.DataFrame:
    """Downloads candles from Yahoo Finance and normalizes them.

    Returns a DataFrame with columns
    ``timestamps, open, high, low, close, volume, amount`` sorted ascending
    by time, without NaNs and with tz-naive timestamps (UTC).
    """
    if interval not in INTERVAL_FREQ:
        raise ValueError(
            f"Unsupported interval: {interval!r}. "
            f"Use one of: {sorted(INTERVAL_FREQ)}"
        )

    raw = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError(
            f"Yahoo Finance returned no data for {ticker!r} "
            f"(interval={interval}, period={period}). "
            "Check the ticker or widen the period."
        )

    df = _normalize(raw)
    if len(df) < 2:
        raise ValueError(f"Not enough data for {ticker!r}: {len(df)} candles.")
    return df


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalizes the raw yfinance DataFrame to the app schema."""
    df = raw.copy()

    # yfinance returns MultiIndex columns when downloading several tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"adj close": "adj_close"})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns in downloaded data: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    # amount = volume * approximate mean price
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

    # Timestamps: tz-naive UTC index to avoid issues with .dt and torch
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    df.index.name = "timestamps"

    df = df[["open", "high", "low", "close", "volume", "amount"]]
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df.reset_index()
