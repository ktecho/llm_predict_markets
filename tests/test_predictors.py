"""Unit tests for the model registry and the predictors.

These tests neither download weights nor hit the network: TimesFM 3.0 is
exercised through a fake forecaster, Kronos and Chronos-2 through fake
vendored predictors, and the registry is checked for all four backends.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import (
    MODEL_REGISTRY,
    Chronos2Predictor,
    KronosPredictor,
    TimesFMPredictor,
)

OHLCV_COLS = ["open", "high", "low", "close", "volume", "amount"]


def _frame(n: int = 96) -> pd.DataFrame:
    """Synthetic OHLCV context (no timestamps column, like ``predict`` input)."""
    base = np.arange(n) + 100.0
    return pd.DataFrame({
        "open": base,
        "high": base + 4.0,
        "low": base - 4.0,
        "close": base + 1.0,
        "volume": np.full(n, 1000.0),
        "amount": np.full(n, 1e5),
    })


class _FakeForecaster:
    """Mimics ``TimesFM3Forecaster.predict_batch`` returning (5, horizon)."""

    def predict_batch(self, contexts, horizon, **kwargs):
        # contexts: list of (num_variates, context_len) arrays
        import dataclasses

        @dataclasses.dataclass
        class _Out:
            forecast: np.ndarray
            quantiles: np.ndarray | None = None
            ts_id: str | None = None

        for ctx in contexts:
            ctx = np.asarray(ctx)
            if ctx.ndim == 1:
                ctx = ctx[None, :]
            n_vars = ctx.shape[0]
            pred = np.stack([
                ctx[i, -1] * (1 + 0.01 * np.arange(horizon, dtype=np.float64))
                for i in range(n_vars)
            ])
            # Mimic (num_variates, horizon) forecast from TimesFM 3.0
            yield _Out(forecast=pred)


def test_timesfm_predictor_candles():
    p = TimesFMPredictor(_FakeForecaster(), max_context=2048)
    x_timestamp = pd.date_range("2024-01-01", periods=96, freq="1h")
    y_timestamp = pd.date_range("2024-01-05", periods=12, freq="1h")
    out = p.predict(_frame(96), x_timestamp, y_timestamp, pred_len=12)

    assert list(out.columns) == OHLCV_COLS
    assert len(out) == 12
    assert isinstance(out.index, pd.DatetimeIndex)
    # Candle geometry is reconciled.
    assert (out["high"] >= out["open"]).all()
    assert (out["high"] >= out["close"]).all()
    assert (out["low"] <= out["open"]).all()
    assert (out["low"] <= out["close"]).all()
    assert (out["volume"] >= 0).all()


def test_registry_has_all_backends():
    backends = {cfg.backend for cfg in MODEL_REGISTRY.values()}
    assert backends == {"timesfm", "moirai", "kronos", "chronos2"}
    assert "3.0" in MODEL_REGISTRY
    assert "moirai-small" in MODEL_REGISTRY
    assert "moirai-base" in MODEL_REGISTRY
    assert "kronos-mini" in MODEL_REGISTRY
    assert "kronos-small" in MODEL_REGISTRY
    assert "kronos-base" in MODEL_REGISTRY
    assert "chronos2" in MODEL_REGISTRY
    assert MODEL_REGISTRY["3.0"].backend == "timesfm"
    assert MODEL_REGISTRY["3.0"].hf_model_id == "google/timesfm-3.0-pytorch"
    assert MODEL_REGISTRY["kronos-small"].backend == "kronos"
    assert MODEL_REGISTRY["chronos2"].backend == "chronos2"
    assert MODEL_REGISTRY["chronos2"].hf_model_id == "amazon/chronos-2"


class _FakeKronosPredictor:
    """Mimics the vendored ``model.KronosPredictor`` (returns raw candles)."""

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T, top_p,
                sample_count, verbose):
        # Deliberately broken geometry so the wrapper must reconcile it.
        open_ = df["open"].to_numpy()[-1] + np.arange(pred_len)
        idx = pd.DatetimeIndex(y_timestamp)
        return pd.DataFrame({
            "open": open_,
            "high": open_ - 10,          # too low  -> pushed up to open/close
            "low": open_ + 10,           # too high -> pushed down to open/close
            "close": open_,
            "volume": np.full(pred_len, -5.0),  # negative -> clamped to 0
        }, index=idx)


def test_kronos_predictor_reconciles_candles():
    p = KronosPredictor(_FakeKronosPredictor(), max_context=512)
    x_timestamp = pd.date_range("2024-01-01", periods=8, freq="1h")
    y_timestamp = pd.date_range("2024-01-02", periods=5, freq="1h")
    out = p.predict(_frame(8), x_timestamp, y_timestamp, pred_len=5,
                    temperature=1.0, top_p=0.9, sample_count=3)

    assert list(out.columns) == OHLCV_COLS
    assert len(out) == 5
    assert isinstance(out.index, pd.DatetimeIndex)
    # Candle geometry is reconciled.
    assert (out["high"] >= out["open"]).all()
    assert (out["high"] >= out["close"]).all()
    assert (out["low"] <= out["open"]).all()
    assert (out["low"] <= out["close"]).all()
    assert (out["volume"] >= 0).all()


class _FakeChronos2Pipeline:
    """Mimics ``chronos.Chronos2Pipeline.predict_quantiles``.

    Returns deliberately broken candle geometry (high below open, low above
    close, negative volume) so the wrapper must reconcile it.
    """

    def predict_quantiles(self, inputs, prediction_length, quantile_levels):
        series = inputs[0]  # (n_variates, history_length)
        n_variates, h = series.shape[0], prediction_length
        t = np.arange(h, dtype=np.float64)
        rows = []
        for i in range(n_variates):
            b = float(series[i, -1])
            if i == 0:      # open
                rows.append(b + t)
            elif i == 1:    # high (too low -> must be pushed up)
                rows.append(b * 0.5 + t)
            elif i == 2:    # low (too high -> must be pushed down)
                rows.append(b * 2.0 + t)
            elif i == 3:    # close
                rows.append(b + t + 1.0)
            else:           # volume (negative -> clamped to 0)
                rows.append(np.full(h, -3.0))
        out = np.stack(rows)  # (n_variates, h)
        quantiles = np.repeat(out[:, :, None], len(quantile_levels), axis=2)
        mean = out
        return [quantiles], [mean]


def test_chronos2_predictor_reconciles_candles():
    p = Chronos2Predictor(_FakeChronos2Pipeline(), max_context=2048)
    x_timestamp = pd.date_range("2024-01-01", periods=8, freq="1h")
    y_timestamp = pd.date_range("2024-01-02", periods=5, freq="1h")
    out = p.predict(_frame(8), x_timestamp, y_timestamp, pred_len=5)

    assert list(out.columns) == OHLCV_COLS
    assert len(out) == 5
    assert isinstance(out.index, pd.DatetimeIndex)
    # Candle geometry is reconciled.
    assert (out["high"] >= out["open"]).all()
    assert (out["high"] >= out["close"]).all()
    assert (out["low"] <= out["open"]).all()
    assert (out["low"] <= out["close"]).all()
    assert (out["volume"] >= 0).all()


def test_load_predictor_unknown_raises():
    from src.models import load_predictor

    with pytest.raises(ValueError):
        load_predictor("definitely-not-a-model")