"""Model registry and predictor loading for TimesFM, Moirai, Kronos and Chronos-2.

Four time-series foundation models are supported, selected from the UI:

- **TimesFM 3.0** (``google/timesfm-3.0-pytorch``, 330M, non-commercial): a
  multivariate foundation model. Unlike 2.5 (univariate), it forecasts all five
  OHLCV variates jointly in one forward pass via variate attention, with native
  support for up to 15360 context length.
- **Moirai** (``Salesforce/moirai-1.1-R-*``, 14M/91M/311M, CC-BY-NC-4.0): a
  truly multivariate probabilistic transformer that forecasts all variates
  jointly in one forward pass, capturing cross-series correlation. Loaded via
  the ``uni2ts`` package.
- **Kronos** (``NeoQuasar/Kronos-{mini,small,base}``, 4.1M/24.7M/102.3M,
  Apache-2.0): a generative time-series foundation model vendored under
  ``vendor/Kronos`` (no PyPI package). It produces candles by token sampling
  controlled by ``temperature``/``top_p``/``sample_count`` and averages the
  sampled paths into a point forecast.
- **Chronos-2** (``amazon/chronos-2``, 120M, Apache-2.0, from
  amazon-science/chronos-forecasting): Amazon's quantile-based encoder-only
  model. Like Moirai, ``Chronos2Predictor`` feeds the five OHLCV variates as
  one multivariate target; the 0.5 quantile is the point forecast.

All backends expose the same ``predict(df, x_timestamp, y_timestamp, pred_len,
verbose, temperature, top_p, sample_count)`` interface and reconcile candle
geometry so ``high >= max(open, close)``, ``low <= min(open, close)`` and
``volume >= 0``.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# TimesFM 3.0 uses input_patch_len=32 (and output_patch_len=64).
# The forecaster handles padding/stitching internally, but the UI still steps
# lookback by 32 for consistency with the original patch constraint.
PATCH_LEN = 32

# OHLCV variates shared by all backends (column order).
OHLCV = ["open", "high", "low", "close", "volume"]

# Kronos has no PyPI package; the repo is vendored under vendor/Kronos.
VENDOR_KRONOS_PATH = Path(__file__).resolve().parent.parent / "vendor" / "Kronos"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str  # "timesfm" | "moirai" | "kronos" | "chronos2"
    hf_model_id: str
    max_context: int
    max_horizon: int
    params: str
    description: str
    num_samples: int | None = None  # only used by Moirai (probabilistic)
    hf_tokenizer_id: str | None = None  # only used by Kronos (vendored models)


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "3.0": ModelConfig(
        name="3.0",
        backend="timesfm",
        hf_model_id="google/timesfm-3.0-pytorch",
        max_context=2048,
        max_horizon=512,
        params="330M",
        description="TimesFM 3.0 (330M, non-commercial). Native multivariate; "
                    "all OHLCV variates forecast jointly in one pass (16k context supported).",
    ),
    "moirai-small": ModelConfig(
        name="moirai-small",
        backend="moirai",
        hf_model_id="Salesforce/moirai-1.1-R-small",
        max_context=512,
        max_horizon=256,
        params="14M",
        num_samples=20,
        description="Moirai 1.1 small (14M, CC-BY-NC). True multivariate "
                    "probabilistic; all OHLCV variates together. Fast on CPU.",
    ),
    "moirai-base": ModelConfig(
        name="moirai-base",
        backend="moirai",
        hf_model_id="Salesforce/moirai-1.1-R-base",
        max_context=512,
        max_horizon=256,
        params="91M",
        num_samples=10,
        description="Moirai 1.1 base (91M, CC-BY-NC). Multivariate probabilistic; "
                    "quality/speed balance.",
    ),
    "moirai-large": ModelConfig(
        name="moirai-large",
        backend="moirai",
        hf_model_id="Salesforce/moirai-1.1-R-large",
        max_context=512,
        max_horizon=256,
        params="311M",
        num_samples=10,
        description="Moirai 1.1 large (311M, CC-BY-NC). Best quality, slow on CPU.",
    ),
    "kronos-mini": ModelConfig(
        name="kronos-mini",
        backend="kronos",
        hf_model_id="NeoQuasar/Kronos-mini",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-2k",
        max_context=2048,
        max_horizon=512,
        params="4.1M",
        description="Kronos mini (4.1M, Apache-2.0). Generative; token sampling. "
                    "Lightest and ideal for CPU, with the longest context (2048).",
    ),
    "kronos-small": ModelConfig(
        name="kronos-small",
        backend="kronos",
        hf_model_id="NeoQuasar/Kronos-small",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        max_horizon=256,
        params="24.7M",
        description="Kronos small (24.7M, Apache-2.0). Generative; sampled point "
                    "forecast (temperature/top-p/sample count). Balanced.",
    ),
    "kronos-base": ModelConfig(
        name="kronos-base",
        backend="kronos",
        hf_model_id="NeoQuasar/Kronos-base",
        hf_tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
        max_context=512,
        max_horizon=256,
        params="102.3M",
        description="Kronos base (102.3M, Apache-2.0). Generative; best quality, "
                    "slower on CPU.",
    ),
    "chronos2": ModelConfig(
        name="chronos2",
        backend="chronos2",
        hf_model_id="amazon/chronos-2",
        max_context=2048,
        max_horizon=1024,
        params="120M",
        description="Chronos-2 (120M, Apache-2.0, Amazon Science). Quantile-based; "
                    "all OHLCV variates forecast jointly.",
    ),
}

DEFAULT_MODEL = "moirai-base"


def device_details(device: str) -> str:
    """Human-readable description of a compute device (hardware, driver)."""
    try:
        import torch

        if device == "cpu":
            return f"CPU ({torch.get_num_threads()} torch threads)"
        if device == "cuda":
            name = torch.cuda.get_device_name(0)
            return f"NVIDIA GPU (CUDA): {name}"
        if device == "xpu":
            try:
                props = torch.xpu.get_device_properties(0)
                driver = getattr(props, "driver_version", None) or "unknown"
                return (
                    f"Intel GPU (XPU): {torch.xpu.get_device_name(0)}, "
                    f"driver {driver}"
                )
            except Exception:
                return "Intel GPU (XPU)"
        if device == "npu":
            try:
                from openvino import Core

                core = Core()
                name = next(
                    (
                        core.get_property(d, "FULL_DEVICE_NAME")
                        for d in core.available_devices
                        if d == "NPU"
                    ),
                    None,
                )
                if name:
                    return f"Intel NPU (OpenVINO): {name}"
                return "Intel NPU (OpenVINO)"
            except Exception:
                return "Intel NPU (OpenVINO)"
        if device == "mps":
            return "Apple Silicon GPU (MPS)"
    except Exception:
        pass
    return device


def _openvino_npu_available() -> bool:
    """True when the Intel NPU is reachable through OpenVINO.

    Intel NPUs (AI Boost on Core Ultra) have no native PyTorch ``torch.device``;
    they are reached through the OpenVINO runtime (``openvino`` package). We ask
    OpenVINO ``Core`` for its available devices and look for an ``NPU`` device.
    """
    try:
        from openvino import Core

        core = Core()
        return "NPU" in core.available_devices
    except Exception:
        return False


def available_devices() -> list[str]:
    """Compute devices available in this environment.

    - ``cpu``: always available.
    - ``cuda``: a CUDA-capable NVIDIA GPU (torch built with CUDA).
    - ``xpu``: Intel GPU (Arc / Iris Xe) via torch XPU backend. On Linux
      requires a torch XPU build (download.pytorch.org/whl/xpu) and,
      depending on the GPU, intel-extension-for-pytorch.
    - ``npu``: Intel NPU (AI Boost on Core Ultra) via OpenVINO. Requires the
      ``openvino`` package and the Intel NPU driver; no native torch device.
    - ``mps``: Apple Silicon.

    Note: TimesFM / Moirai / Chronos-2 torch inference only accelerates on a
    CUDA GPU and falls back to CPU otherwise; Kronos can additionally use XPU,
    MPS or the NPU when present (see :func:`effective_device`, and
    :func:`_compile_kronos_for_npu` for the NPU path). The UI offers whatever
    this function detects.
    """
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            devices.append("xpu")
        if _openvino_npu_available():
            devices.append("npu")
        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    logger.debug(
        "Available compute devices: %s",
        {d: device_details(d) for d in devices},
    )
    return devices


def _build_candles(
    f_open: np.ndarray,
    f_high: np.ndarray,
    f_low: np.ndarray,
    f_close: np.ndarray,
    f_volume: np.ndarray,
    y_timestamp,
) -> pd.DataFrame:
    """Reconciles candle geometry and returns the app schema DataFrame.

    Ensures ``high >= max(open, close)``, ``low <= min(open, close)`` and
    ``volume >= 0``, and computes ``amount = volume * mean price``. The result
    is indexed by the future ``y_timestamp`` with columns
    ``open, high, low, close, volume, amount``.
    """
    high = np.maximum(f_high, np.maximum(f_open, f_close))
    low = np.minimum(f_low, np.minimum(f_open, f_close))
    volume = np.maximum(f_volume, 0.0)
    amount = volume * (f_open + high + low + f_close) / 4.0

    index = pd.DatetimeIndex(y_timestamp)
    index.name = "timestamps"
    return pd.DataFrame(
        {
            "open": f_open,
            "high": high,
            "low": low,
            "close": f_close,
            "volume": volume,
            "amount": amount,
        },
        index=index,
    )


class TimesFMPredictor:
    """Wraps TimesFM 3.0 and exposes a predictor-compatible ``predict()`` interface.

    TimesFM 3.0 is a **native multivariate** foundation model (330M, non-commercial).
    It forecasts all five OHLCV variates jointly via variate attention in a single
    forward pass, capturing cross-series correlation. The median quantile (0.5) is
    used as the point forecast; candle geometry is reconciled afterwards:

    - ``open`` / ``close``: TimesFM 3.0 point forecasts.
    - ``high`` = max(hi_forecast, open, close).
    - ``low``  = min(lo_forecast, open, close).
    - ``volume``: TimesFM forecast, clamped at >= 0.
    - ``amount`` = volume * mean price.

    The returned DataFrame is indexed by the future timestamps, with columns
    ``open, high, low, close, volume, amount``.
    """

    def __init__(self, forecaster, max_context: int):
        self.forecaster = forecaster
        self.max_context = max_context

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int = 120,
        verbose: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        """Predicts ``pred_len`` candles from the OHLCV context ``df``.

        ``temperature``/``top_p``/``sample_count`` are accepted for a uniform
        interface across backends but are ignored (TimesFM is deterministic,
        quantile-based).
        """
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}.")
        if len(df) < 2:
            raise ValueError("At least 2 context candles are required.")

        # Multivariate target: (num_variates=5, context_length)
        target = np.stack(
            [df[c].to_numpy(dtype=np.float32) for c in OHLCV]
        )
        outputs = list(
            self.forecaster.predict_batch(
                contexts=[target],
                horizon=int(pred_len),
            )
        )
        forecast = outputs[0].forecast
        if forecast is None:
            raise RuntimeError("TimesFM 3.0 returned no forecast.")
        pred = np.asarray(forecast, dtype=np.float64)  # shape (5, pred_len)

        # Handle both (5, pred_len) and (pred_len,) edge cases
        if pred.ndim == 1:
            # Should not happen for multivariate (5 variates), but guard
            raise RuntimeError(
                f"Unexpected univariate forecast shape {pred.shape}; "
                f"expected (5, {pred_len})."
            )
        f_open, f_high, f_low, f_close, f_volume = pred
        return _build_candles(f_open, f_high, f_low, f_close, f_volume, y_timestamp)


class MoiraiPredictor:
    """Wraps a Moirai module and exposes the app ``predict()`` interface.

    Moirai is a truly multivariate probabilistic transformer: all OHLCV
    variates are fed as one series (``target_dim = 5``) and forecast jointly in
    a single forward pass, so the cross-series relationship is learned by the
    model. We use the median over ``num_samples`` sampled trajectories as the
    point forecast and reconcile candle geometry (``high >= max(open, close)``,
    ``low <= min(open, close)``, ``volume >= 0``) as a safety net.
    """

    def __init__(self, module, max_context: int, num_samples: int = 20,
                 patch_size: str = "auto"):
        self.module = module
        self.max_context = max_context
        self.num_samples = num_samples
        self.patch_size = patch_size

    def predict(self, df, x_timestamp, y_timestamp, pred_len=120, verbose=False,
                temperature=1.0, top_p=0.9, sample_count=1):
        """Forecast all OHLCV variates jointly via a Moirai predictive head.

        ``temperature``/``top_p``/``sample_count`` are accepted for a uniform
        interface across backends but unused here (sampling is controlled by the
        registry's ``num_samples``).
        """
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}.")
        if len(df) < 2:
            raise ValueError("At least 2 context candles are required.")

        from gluonts.dataset.pandas import PandasDataset
        from uni2ts.model.moirai import MoiraiForecast

        wide = df[OHLCV].copy()
        wide.index = pd.DatetimeIndex(x_timestamp)
        ds = PandasDataset(wide, target=OHLCV)

        model = MoiraiForecast(
            module=self.module,
            prediction_length=int(pred_len),
            context_length=int(len(wide)),
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=len(OHLCV),
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
        predictor = model.create_predictor(batch_size=1)
        forecast = next(iter(predictor.predict(ds)))
        median = np.asarray(forecast.median, dtype=np.float64)  # (pred_len, 5)

        f_open = median[:, 0]
        f_high = median[:, 1]
        f_low = median[:, 2]
        f_close = median[:, 3]
        f_volume = median[:, 4]

        return _build_candles(f_open, f_high, f_low, f_close, f_volume, y_timestamp)


def _ensure_kronos_importable() -> None:
    """Adds vendor/Kronos to sys.path so `from model import ...` works."""
    if not VENDOR_KRONOS_PATH.is_dir():
        raise FileNotFoundError(
            f"Vendored Kronos repo not found at {VENDOR_KRONOS_PATH}. "
            "Clone it with: git clone "
            "https://github.com/shiyu-coder/Kronos vendor/Kronos"
        )
    path = str(VENDOR_KRONOS_PATH)
    if path not in sys.path:
        sys.path.insert(0, path)


class KronosPredictor:
    """Wraps the vendored NeoQuasar Kronos predictor for the app ``predict()``.

    Kronos is a generative, token-based foundation model. It produces candle
    sequences by sampling from the next-token distribution, controlled by
    ``temperature``, ``top_p`` and ``sample_count`` (the sampled paths are
    averaged). It is not published on PyPI, so it is loaded from the vendored
    ``vendor/Kronos`` repo (see ``_ensure_kronos_importable``).

    The wrapped predictor returns a DataFrame indexed by the future timestamps
    with OHLCV columns; this wrapper still reconciles candle geometry
    (``high >= max(open, close)``, ``low <= min(open, close)``, ``volume >= 0``)
    as a safety net.
    """

    def __init__(self, predictor, max_context: int):
        self.predictor = predictor
        self.max_context = max_context

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int = 120,
        verbose: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}.")
        if len(df) < 2:
            raise ValueError("At least 2 context candles are required.")

        pred = self.predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=verbose,
        )
        return _reconcile_kronos_candles(pred, y_timestamp)


def _reconcile_kronos_candles(pred: pd.DataFrame, y_timestamp) -> pd.DataFrame:
    """Reconciles the vendored Kronos output into the app candle schema."""
    out = pred.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    open_ = out["open"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    volume = out["volume"].to_numpy(dtype=np.float64)

    return _build_candles(open_, high, low, close, volume, y_timestamp)


class Chronos2Predictor:
    """Wraps Amazon's Chronos-2 pipeline and exposes the app ``predict()``.

    Chronos-2 (``amazon/chronos-2``, 120M, Apache-2.0) is a quantile-based,
    encoder-only model from the ``chronos-forecasting`` PyPI package. Like
    Moirai we feed the five OHLCV variates as one multivariate target
    ``(5, context_length)`` and forecast them jointly; the 0.5 quantile is
    used as the point forecast and candle geometry is reconciled as a safety
    net. ``temperature``/``top_p``/``sample_count`` are accepted for a uniform
    interface but unused (Chronos-2 is quantile-based, not sample-based).
    """

    def __init__(self, pipeline, max_context: int):
        self.pipeline = pipeline
        self.max_context = max_context

    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int = 120,
        verbose: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}.")
        if len(df) < 2:
            raise ValueError("At least 2 context candles are required.")

        # One multivariate target: (n_variates=5, history_length).
        inputs = np.stack(
            [df[c].to_numpy(dtype=np.float32) for c in OHLCV]
        )
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs=[inputs],
            prediction_length=int(pred_len),
            quantile_levels=[0.5],
        )
        # quantiles[0] has shape (n_variates, pred_len, len(quantile_levels)).
        pred = np.asarray(quantiles[0][..., 0], dtype=np.float64)  # (5, pred_len)

        f_open, f_high, f_low, f_close, f_volume = pred
        return _build_candles(f_open, f_high, f_low, f_close, f_volume, y_timestamp)


def _load_timesfm(cfg: ModelConfig, device: str):
    from timesfm3 import ModelConfig as TimesFM3Config, TimesFM3Forecaster  # noqa: PLC0415

    tcfg = TimesFM3Config(
        checkpoint_path=cfg.hf_model_id,
        device=device,
        per_core_batch_size=32,
    )
    forecaster = TimesFM3Forecaster(tcfg)
    return TimesFMPredictor(forecaster, max_context=cfg.max_context)


def _load_moirai(cfg: ModelConfig, device: str):
    from uni2ts.model.moirai import MoiraiModule

    module = MoiraiModule.from_pretrained(cfg.hf_model_id)
    module.eval()
    return MoiraiPredictor(
        module,
        max_context=cfg.max_context,
        num_samples=cfg.num_samples or 20,
    )


def _compile_kronos_for_npu(model):
    """Compiles a Kronos model for the Intel NPU via OpenVINO's torch backend.

    The Intel NPU has no ``torch.device`` string, so the compiled module still
    runs on CPU tensors while OpenVINO offloads the operators to the NPU. This
    is the experimental path documented in the README; if OpenVINO's torch
    backend is missing it falls back to the original CPU model.
    """
    try:
        import torch

        backend = getattr(torch.backends, "openvino", None)
        if backend is None or not getattr(backend, "is_available", lambda: False)():
            raise RuntimeError("OpenVINO torch backend not available")
        compiled = torch.compile(model, backend="openvino")
        logger.info("Kronos compiled for the Intel NPU via OpenVINO.")
        return compiled
    except Exception as e:  # noqa: BLE001 - degrade gracefully to CPU
        logger.warning(
            "Intel NPU (OpenVINO) compile failed (%s); falling back to CPU.", e
        )
        return model


def _load_kronos(cfg: ModelConfig, device: str):
    _ensure_kronos_importable()
    from model import (  # noqa: PLC0415
        Kronos,
        KronosPredictor as _VendoredKronosPredictor,
        KronosTokenizer,
    )

    tokenizer = KronosTokenizer.from_pretrained(cfg.hf_tokenizer_id)
    model = Kronos.from_pretrained(cfg.hf_model_id)

    # The Intel NPU has no torch device string; OpenVINO compiles the model but
    # it still runs on CPU tensors, so the vendored predictor (which calls
    # ``model.to(device)``) must be handed ``cpu``.
    if device == "npu":
        model = _compile_kronos_for_npu(model)
        device = "cpu"

    predictor = _VendoredKronosPredictor(
        model,
        tokenizer,
        device=device,
        max_context=cfg.max_context,
    )
    predictor.eval = getattr(model, "eval", lambda: None)  # ensure eval mode
    return KronosPredictor(predictor, max_context=cfg.max_context)


def _load_chronos2(cfg: ModelConfig, device: str):
    from chronos import Chronos2Pipeline

    kwargs = {"device_map": device} if device != "cpu" else {}
    pipeline = Chronos2Pipeline.from_pretrained(cfg.hf_model_id, **kwargs)
    return Chronos2Predictor(pipeline, max_context=cfg.max_context)


# Devices that only Kronos can consume natively. The torch-based backends
# (TimesFM / Moirai / Chronos-2) have no accelerated path on an Intel XPU, an
# Intel NPU or an Apple MPS device, so they must fall back to CPU. Kronos runs
# on XPU/MPS; on the NPU it runs through OpenVINO (see _load_kronos).
_KRONOS_ONLY_DEVICES = frozenset({"xpu", "npu", "mps"})


def effective_device(backend: str, device: str) -> str:
    """Returns the device a backend should actually run on.

    ``xpu`` (Intel GPU), ``npu`` (Intel NPU, OpenVINO) and ``mps`` (Apple
    Silicon) only have a working path in Kronos; TimesFM, Moirai and Chronos-2
    fall back to ``cpu`` on them (their torch inference only accelerates on
    CUDA). ``cpu`` and ``cuda`` always pass through unchanged.
    """
    if backend != "kronos" and device in _KRONOS_ONLY_DEVICES:
        return "cpu"
    return device


@lru_cache(maxsize=8)
def load_predictor(model_name: str = DEFAULT_MODEL, device: str = "cpu"):
    """Loads (cached) the selected model's predictor.

    Dispatches to TimesFM, Moirai, Kronos or Chronos-2 based on the registry
    backend. The first call downloads the weights from HuggingFace; subsequent
    calls reuse it.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    cfg = MODEL_REGISTRY[model_name]

    requested = device
    device = effective_device(cfg.backend, device)
    if device != requested:
        logger.warning(
            "%s has no %s path; falling back to %s.",
            cfg.backend, requested, device,
        )

    import torch

    logger.info(
        "Initializing %s-%s (%s params) | model=%s device=%s",
        cfg.backend, model_name, cfg.params, cfg.hf_model_id, device,
    )
    logger.info("torch %s | inference device: %s -> %s",
                torch.__version__, device, device_details(device))
    torch.set_float32_matmul_precision("high")

    t0 = time.time()
    if cfg.backend == "timesfm":
        predictor = _load_timesfm(cfg, device)
    elif cfg.backend == "moirai":
        predictor = _load_moirai(cfg, device)
    elif cfg.backend == "kronos":
        predictor = _load_kronos(cfg, device)
    elif cfg.backend == "chronos2":
        predictor = _load_chronos2(cfg, device)
    else:
        raise ValueError(f"Unknown backend: {cfg.backend!r} for {model_name!r}.")
    logger.info(
        "%s-%s ready on %s in %.1fs (max_context=%d)",
        cfg.backend, model_name, device, time.time() - t0, cfg.max_context,
    )
    return predictor