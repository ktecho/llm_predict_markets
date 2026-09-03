"""Streamlit app: price prediction with TimesFM + Yahoo Finance.

Run with:  uv run streamlit run app.py
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.backtest import compute_metrics
from src.data import (
    INTERVAL_FREQ,
    MAX_PERIOD_BY_INTERVAL,
    VALID_PERIODS,
    download_ohlcv,
)
from src.models import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    available_devices,
    device_details,
    load_predictor,
)
from src.plotting import backtest_figure, forecast_figure
from src.predict import backtest, forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="TimesFM · Moirai · Kronos · Chronos-2 Price Predictor", layout="wide")
st.title("📈 Foundation-model Price Predictor")
st.caption(
    "OHLCV candle prediction with Google [TimesFM]"
    "(https://github.com/google-research/timesfm), Salesforce [Moirai]"
    "(https://github.com/SalesforceAIResearch/uni2ts), [Kronos]"
    "(https://github.com/shiyu-coder/Kronos) or Amazon [Chronos-2]"
    "(https://github.com/amazon-science/chronos-forecasting) and Yahoo Finance "
    "data. For research purposes only — not financial advice."
)


def model_label(name: str) -> str:
    """Human-readable label for a model in the registry."""
    cfg = MODEL_REGISTRY[name]
    if cfg.backend == "timesfm":
        return f"TimesFM-{cfg.name} ({cfg.params})"
    if cfg.backend == "kronos":
        return f"Kronos {cfg.name.split('-', 1)[1]} ({cfg.params})"
    if cfg.backend == "chronos2":
        return f"Chronos-2 ({cfg.params})"
    # moirai-<size> -> "Moirai <size>"
    size = cfg.name.split("-", 1)[1]
    return f"Moirai {size} ({cfg.params})"


@st.cache_resource(show_spinner=False)
def get_predictor(model_name: str, device: str):
    return load_predictor(model_name, device=device)


@st.cache_data(ttl=900, show_spinner=False)
def get_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    return download_ohlcv(ticker, interval, period)


@st.cache_resource(show_spinner=False)
def get_available_devices() -> list[str]:
    """Detected once per server process so startup logs do not repeat."""
    devices = available_devices()
    logger.info(
        "Compute devices detected: %s",
        {d: device_details(d) for d in devices},
    )
    return devices


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.header("Data")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="BTC-USD",
                           help="E.g.: BTC-USD, AAPL, ^GSPC, EURUSD=X")
    interval = st.selectbox("Timeframe", list(INTERVAL_FREQ), index=7)
    max_period = MAX_PERIOD_BY_INTERVAL.get(interval, "max")
    # VALID_PERIODS is ordered from shortest to longest duration
    max_idx = VALID_PERIODS.index(max_period) if max_period in VALID_PERIODS else len(VALID_PERIODS) - 1
    period_options = VALID_PERIODS[: max_idx + 1]
    period = st.selectbox("Historical period", period_options,
                          index=len(period_options) - 1)

    st.header("Model")
    model_name = st.selectbox(
        "Foundation model",
        list(MODEL_REGISTRY),
        index=list(MODEL_REGISTRY).index(DEFAULT_MODEL),
        format_func=model_label,
    )
    cfg = MODEL_REGISTRY[model_name]
    st.caption(cfg.description)

    devices = get_available_devices()
    device = st.selectbox(
        "Compute device", devices, index=0,
        format_func=device_details,
        help="TimesFM/Moirai/Chronos-2 accelerate on a CUDA (NVIDIA) GPU; "
             "XPU (Intel GPU)/NPU (OpenVINO)/MPS fall back to CPU for them, "
             "but Kronos can use XPU/MPS natively (NPU via OpenVINO). "
             "If unsure, use cpu.",
    )

    mode = st.radio("Mode", ["Forecast", "Backtest"], horizontal=True)

    lookback_default = min(384, cfg.max_context)
    lookback = st.slider("Lookback (context candles)", 64, cfg.max_context,
                         lookback_default, step=32)
    if mode == "Forecast":
        pred_len = st.slider("Candles to predict", 8, 240, 120, step=8)
    else:
        pred_len = st.slider("Backtest candles", 8, 240, 60, step=8)

    # Sampling controls only affect Kronos (generative token sampling).
    if cfg.backend == "kronos":
        st.header("Sampling")
        temperature = st.slider("Temperature (T)", 0.1, 2.0, 1.0, step=0.1)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
        sample_count = st.slider(
            "Sample count", 1, 20, 1,
            help="Forecast paths generated and averaged. More samples = "
                 "smoother prediction, but proportionally slower.",
        )
    else:
        temperature, top_p, sample_count = 1.0, 0.9, 1

    run = st.button("🚀 Predict", type="primary", width="stretch")

# -------------------------------------------------------------------- main ---
try:
    with st.spinner(f"Downloading {ticker} ({interval}, {period})…"):
        df = get_data(ticker, interval, period)
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

needed = lookback + (pred_len if mode == "Backtest" else 0)
st.info(
    f"**{ticker}** · {interval} · {len(df)} candles downloaded "
    f"({df['timestamps'].iloc[0]:%Y-%m-%d %H:%M} → {df['timestamps'].iloc[-1]:%Y-%m-%d %H:%M} UTC) · "
    f"last close: **{df['close'].iloc[-1]:.4g}**"
)
if len(df) < needed:
    st.warning(
        f"At least {needed} candles are needed for lookback={lookback}"
        + (" + backtest" if mode == "Backtest" else "")
        + f" but only {len(df)} are available. Widen the period or reduce the lookback."
    )
    st.stop()

if not run:
    st.plotly_chart(forecast_figure(df.tail(min(lookback, len(df))), df.tail(0),
                                    title=f"{ticker} · history"),
                    width="stretch")
    st.stop()

with st.spinner(f"Loading {model_label(model_name)} model (first run downloads from HuggingFace)…"):
    logger.info("Predictor requested: model=%s device=%s", model_name, device)
    try:
        predictor = get_predictor(model_name, device)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

if mode == "Forecast":
    with st.spinner(f"Generating prediction of {pred_len} candles…"):
        try:
            pred_df = forecast(
                predictor, df, interval,
                pred_len=pred_len, lookback=lookback,
                verbose=True,
                temperature=temperature, top_p=top_p, sample_count=sample_count,
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    st.plotly_chart(
        forecast_figure(df.tail(lookback), pred_df,
                        title=f"{ticker} · {interval} · {model_label(model_name)}"),
        width="stretch",
    )
    st.subheader("Predicted candles")
    st.dataframe(pred_df.round(4), width="stretch")
    st.download_button(
        "⬇️ Download prediction (CSV)",
        pred_df.to_csv().encode(),
        file_name=f"{ticker}_{interval}_forecast_{model_name}.csv",
        mime="text/csv",
    )

else:  # Backtest
    with st.spinner(f"Running backtest ({pred_len} candles)…"):
        try:
            pred_df, actual_df = backtest(
                predictor, df, interval,
                pred_len=pred_len, lookback=lookback,
                verbose=True,
                temperature=temperature, top_p=top_p, sample_count=sample_count,
            )
        except Exception as e:
            st.error(f"Backtest error: {e}")
            st.stop()

    context_df = df.iloc[-lookback - pred_len:-pred_len]
    prev_close = float(context_df["close"].iloc[-1])
    metrics = compute_metrics(actual_df["close"], pred_df["close"], prev_close=prev_close)

    cols = st.columns(4)
    cols[0].metric("MAE", f"{metrics['mae']:.4g}")
    cols[1].metric("RMSE", f"{metrics['rmse']:.4g}")
    cols[2].metric("MAPE", f"{metrics['mape']:.2f}%")
    cols[3].metric("Directional accuracy", f"{metrics['directional_accuracy']:.1f}%")

    st.plotly_chart(
        backtest_figure(context_df, actual_df, pred_df,
                        title=f"{ticker} · {interval} · backtest {model_label(model_name)}"),
        width="stretch",
    )

    comparison = pd.DataFrame({
        "timestamp": actual_df.index,
        "actual_close": actual_df["close"].to_numpy(),
        "predicted_close": pred_df["close"].to_numpy(),
    })
    comparison["error_%"] = (comparison["predicted_close"] - comparison["actual_close"]) / comparison["actual_close"] * 100
    st.subheader("Actual vs predicted (close)")
    st.dataframe(comparison.round(4), width="stretch")
    st.download_button(
        "⬇️ Download comparison (CSV)",
        comparison.to_csv(index=False).encode(),
        file_name=f"{ticker}_{interval}_backtest_{model_name}.csv",
        mime="text/csv",
    )
