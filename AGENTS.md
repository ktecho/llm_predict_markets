# AGENTS.md

## Context

Local price prediction app: four interchangeable time-series foundation models —
**TimesFM** (Google, PyPI `timesfm`), **Moirai** (Salesforce, via the `uni2ts`
package), **Kronos** (NeoQuasar, vendored under `vendor/Kronos`, no PyPI
package) and **Chronos-2** (Amazon, via the `chronos-forecasting` package) +
**Yahoo Finance** (yfinance) + **Streamlit** + **Plotly**. Research only, no real
trading.

## Environment and commands

- Package manager: **uv** with **Python 3.11** (the system has 3.14; torch
  comes from an index defined in `pyproject.toml`).
- **Default: CPU-only torch from PyPI** — the CUDA index block is commented
  out.
- `requires-python` is **`>=3.11,<3.12`** (3.11-only on purpose): the NVIDIA
  `cu124` torch wheelhouse only resolves cleanly for CPython 3.11. Keep the
  whole project on 3.11.
- To keep every package at the best available version from all configured
  indexes (PyPI plus any CUDA wheelhouse), `index-strategy =
  "unsafe-best-match"` is set under `[tool.uv]`. Only matters while the CUDA
  index is enabled.
- Install: `uv sync --python 3.11` (installs `timesfm` + `uni2ts` +
  `chronos-forecasting`; `uni2ts` pins torch <2.5, numpy 1.26 and einops 0.7,
  shared with the other backends; `transformers` is pinned `<5` because v5
  needs torch >=2.5).
- Kronos has no PyPI package: clone it into `vendor/` (gitignored) with
  `git clone https://github.com/shiyu-coder/Kronos vendor/Kronos`.
- Run the app: `uv run streamlit run app.py`
- Tests: `uv run pytest` (unit tests, no network or model needed; ~1s)

## Design decisions

- **Four backends** in `src/models.py`, selected via `MODEL_REGISTRY` and a
  `backend` field ("timesfm" | "moirai" | "kronos" | "chronos2");
  `load_predictor()` dispatches on it.
- **TimesFM 3.0** (`google/timesfm-3.0-pytorch`, 330M, non-commercial). A
  **native multivariate** foundation model via variate attention: `TimesFMPredictor`
  feeds all five OHLCV variates as one multivariate target `(5, context_length)`
  and forecasts them jointly in a single forward pass; the 0.5 quantile is the
  point forecast.
- **Moirai 1.1-R** (`Salesforce/moirai-1.1-R-{small,base,large}`), CC-BY-NC-4.0.
  A **truly multivariate probabilistic** model: `MoiraiPredictor` feeds all five
  OHLCV variates as one GluonTS series (`target_dim=5`, via
  `PandasDataset(df, target=OHLCV)`) and forecasts them jointly; the median over
  `num_samples` trajectories is the point forecast.
- **Kronos** (`NeoQuasar/Kronos-{mini,small,base}`, 4.1M/24.7M/102.3M),
  Apache-2.0. A **generative token-based** model vendored under `vendor/Kronos`
  (no PyPI package, gitignored). `KronosPredictor` wraps the vendored predictor
  and adds the app's `predict()` interface; candles come from token sampling
  controlled by `temperature`, `top_p` and `sample_count`, which are passed
  through `forecast()`/`backtest()` (ignored by the other backends).
- **Chronos-2** (`amazon/chronos-2`, 120M), Apache-2.0, from the
  `chronos-forecasting` package. A **quantile-based encoder-only** model:
  `Chronos2Predictor` feeds the five OHLCV variates as one multivariate target
  `(5, context_length)` and uses the 0.5 quantile as the point forecast.
- **Shared candle reconciliation**: every predictor ensures
  `high >= max(open,close)`, `low <= min(open,close)`, `volume >= 0` and
  `amount = volume * mean price`, returning a DataFrame indexed by future
  timestamps with columns `open, high, low, close, volume, amount`.
- **Model registry** (`src/models.py`): `3.0` (timesfm, max_context=2048,
  16k supported, max_horizon=512), `moirai-small`/`base`/`large`
  (max_context=512, num_samples 20/10/10), `kronos-mini`/`small`/`base`
  (max_context 2048/512/512) and `chronos2` (max_context=2048, max_horizon=1024).
  TimesFM uses input patches of 32, so the lookback slider uses multiples of 32.
- **Devices**: CPU by default; GPU if a CUDA (NVIDIA) torch build is installed.
  The UI offers whatever `available_devices()` detects — `cpu`, `cuda`, `xpu`
  (Intel GPU), `npu` (Intel NPU via OpenVINO) and `mps` (Apple Silicon).
  TimesFM/Moirai/Chronos-2 torch inference only accelerates on CUDA, so
  `effective_device()` makes them fall back to CPU on `xpu`/`npu`/`mps`; only
  Kronos can run on XPU/MPS natively (the NPU has no torch device string, so
  `_load_kronos` compiles it with `torch.compile(model, backend="openvino")`
  and still runs on CPU tensors, falling back to CPU if OpenVINO is absent).
  `device_details()` knows all names.
- **Logging**: `logging` INFO messages at startup/model init (`src/models.py`,
  `app.py`) report the torch build and the selected device/model.
- **Data format**: `src/data.py` normalizes yfinance to
  `timestamps, open, high, low, close, volume, amount` (tz-naive UTC, no NaN).
  `amount = volume * mean price` if missing.
- **Future timestamps**: fixed frequency per interval (`INTERVAL_FREQ` in
  `src/data.py`). Documented approximation for markets with trading halts.
- **Streamlit caches**: `@st.cache_resource` for the predictor (HF weights),
  `@st.cache_data(ttl=900)` for downloads.

## Style

- Code and comments in English, simple and modular (src/ by layers).
- Do not add heavy dependencies without a reason; matplotlib is not used (Plotly).
- When changing the `src/` interface, update `app.py`, tests and this file.