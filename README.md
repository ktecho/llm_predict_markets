# TimesFM · Moirai · Kronos · Chronos-2 Price Predictor

Local price prediction app (OHLCV candles) using Google's
[**TimesFM**](https://github.com/google-research/timesfm) (330M, non-commercial),
Salesforce's [**Moirai**](https://github.com/SalesforceAIResearch/uni2ts)
(CC-BY-NC-4.0), [**Kronos**](https://github.com/shiyu-coder/Kronos)
(Apache-2.0) or Amazon's [**Chronos-2**](https://github.com/amazon-science/chronos-forecasting)
(Apache-2.0) plus **Yahoo Finance** data. For research purposes only —
**not financial advice**.

## Features

- Downloads OHLCV series from Yahoo Finance by **ticker** and **timeframe**
  (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk).
- **Four interchangeable foundation models** (sidebar):
  - **TimesFM 3.0** — `330M` params, native **multivariate** (non-commercial).
  - **Moirai 1.1-R** — `small`/`base`/`large` (14M/91M/311M), probabilistic and
    truly **multivariate** (forecasts all OHLCV variates together).
  - **Kronos** — `mini`/`small`/`base` (4.1M/24.7M/102.3M), **generative** with
    sampling controls (temperature, top-p, sample count), Apache-2.0.
  - **Chronos-2** — `120M`, **quantile-based** (Amazon), univariate and
    multivariate, Apache-2.0.
- **Forecast mode**: predicts the next N candles from recent context.
- **Backtest mode**: predicts a known historical window and compares it against
  reality (MAE, RMSE, MAPE, directional accuracy, actual vs predicted chart).
- Interactive candlestick charts (Plotly) and CSV export.
- Runs on **CPU** by default; **NVIDIA GPU (CUDA)** supported if torch is built
  for the GPU. Selectable in the sidebar ("Compute device").

## How prediction works

All four models feed the five OHLCV series (**open, high, low, close, volume**)
and return a forecast for each candle:

- **TimesFM 3.0** is a native **multivariate** model with variate attention; it
  forecasts the five OHLCV series **jointly** in a single forward pass (the 0.5
  quantile is the point forecast).
- **Moirai** is a multivariate probabilistic transformer; it forecasts all five
  variates **jointly** (one series with `target_dim = 5`), so cross-series
  correlation is modeled. The median over sampled trajectories is used as the
  point forecast.
- **Kronos** is a generative token-based model (vendored under `vendor/Kronos`).
  It samples candle sequences from the next-token distribution — controlled by
  **temperature**, **top-p** and **sample count** — and averages the sampled
  paths into the point forecast.
- **Chronos-2** is Amazon's quantile-based encoder-only model. Like Moirai, it
  forecasts all five OHLCV variates **jointly**; the 0.5 quantile is used as
  the point forecast.

Either way each predicted candle is reconciled so the geometry is consistent:
`high = max(hi, open, close)`, `low = min(lo, open, close)`, and
`volume = max(vol, 0)`. `amount = volume * mean price`.

## Installation

Requires [uv](https://docs.astral.sh/uv/) (manages Python 3.11 automatically).
Installation also pulls in `uni2ts` (Moirai), which pins **torch <2.5**, numpy
1.26 and einops 0.7; this is shared with the other backends. `chronos-forecasting`
(Chronos-2) brings `transformers`, which is pinned `<5` because v5 needs
torch >=2.5.

```bash
uv sync --python 3.11   # CPU-only torch + timesfm + uni2ts + chronos-forecasting

# Kronos has no PyPI package; clone it once into vendor/ (gitignored):
git clone https://github.com/shiyu-coder/Kronos vendor/Kronos
```

## Usage

```bash
uv run streamlit run app.py
```

Open http://localhost:8501, pick ticker/timeframe/model and press **Predict**.
The first run of each model downloads its weights from HuggingFace.

Startup and model-loading logs (including the selected inference device) are
printed to the console where Streamlit runs, e.g.:

```
INFO src.models: torch 2.4.1+cpu | inference device: cpu -> CPU (8 torch threads)
INFO src.models: moirai-moirai-base ready on cpu in 7.9s (max_context=512)
```

## Parameters

All parameters are set in the sidebar:

### Data

| Parameter | What it does |
|---|---|
| **Ticker** | Yahoo Finance symbol. Stocks (`AAPL`, `TSLA`), indices (`^GSPC`), forex (`EURUSD=X`), crypto (`BTC-USD`)… |
| **Timeframe** | Candle interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk. Determines the frequency of both the history and the predicted candles. |
| **Historical period** | How much history to download (1d … max). Limited by the timeframe: Yahoo only keeps ~7 days of 1m data, ~1 month of 5m–30m, ~2 years of 1h. |

### Model

| Parameter | What it does |
|---|---|
| **Foundation model** | `TimesFM-3.0` (330M, native multivariate), `Moirai small/base/large` (probabilistic + true multivariate), `Kronos mini/small/base` (generative) or `Chronos-2` (quantile-based, Amazon). Default: `moirai-base`. |
| **Compute device** | Auto-detected from torch and offered in the sidebar: `cpu`, `cuda` (NVIDIA GPU), `xpu` (Intel GPU), `npu` (Intel NPU via OpenVINO) and `mps` (Apple Silicon). TimesFM/Moirai/Chronos-2 only accelerate on `cuda` and fall back to CPU otherwise; **Kronos can also use `xpu`/`mps`, and the `npu` via OpenVINO**. |
| **Mode** | `Forecast`: predicts the next N candles into the future. `Backtest`: predicts a known historical window and compares it against reality with metrics (MAE, RMSE, MAPE, directional accuracy) — use this to judge whether the model works for your ticker/timeframe before trusting a forecast. |
| **Lookback** | Number of past candles fed as context (multiples of 32; up to 2048 for TimesFM/Kronos-mini/Chronos-2, 512 for Moirai/Kronos-small/base). More context = more information, but slower (TimesFM 3.0 supports up to 16k). |
| **Candles to predict / Backtest candles** | Prediction horizon (`pred_len`, max 240): how many candles the model generates. Longer horizons are slower and less reliable. |
| **Temperature (T)** | *(Kronos only)* Sampling temperature: `1.0` is neutral, lower = more conservative, higher = more varied paths. |
| **Top-p** | *(Kronos only)* Nucleus sampling threshold: probability mass kept at each token step. |
| **Sample count** | *(Kronos only)* Forecast paths generated and averaged. More samples = smoother prediction, proportionally slower. |

## Hardware acceleration

The app picks the compute device from the sidebar ("Compute device"). The list
is auto-detected from torch: `cpu` is always available, plus `cuda` (NVIDIA
GPU), `xpu` (Intel GPU), `npu` (Intel NPU, via OpenVINO) and `mps` (Apple
Silicon) when the installed runtime exposes them. Because TimesFM / Moirai /
Chronos-2 torch inference only accelerates on CUDA, selecting `xpu`/`npu`/`mps`
with those backends falls back to CPU; **Kronos is the only backend that can run
natively on Intel XPU / Apple MPS, and on the Intel NPU through OpenVINO**.

By default `pyproject.toml` uses a **CPU-only torch** build from PyPI; the CUDA
index in that file is commented out, so no GPU support is installed.

### NVIDIA GPU (CUDA)

To accelerate inference on an **NVIDIA GPU**, uncomment the CUDA block in
`pyproject.toml` and recreate the environment. Choose the CUDA version that
matches your driver (see `nvidia-smi`); `cu124` is a safe default for recent
NVIDIA drivers and works with the torch `<2.5` pin used here:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
```

Then recreate the environment (this recompiles torch):

```bash
uv sync --python 3.11 --reinstall-package torch
```

Verify that CUDA is active:

```bash
uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

You should see your CUDA version (e.g. `cu124`) and `True`.

### Which CUDA wheel to pick

| PyTorch wheel | Min. NVIDIA driver (Windows) | Notes |
|---|---|---|
| `cu118` | ~450.80 | oldest; for old drivers |
| `cu121` | ~525.60 | |
| `cu124` | ~550.54 | recommended default |

Your concrete CUDA architecture/driver is shown by `nvidia-smi`. TimesFM, Moirai
and Chronos-2 then run inference on the GPU and the sidebar will offer `cuda`.

### Intel GPU (Arc, Iris Xe, Core Ultra iGPU) — XPU backend

PyTorch ships an **XPU** backend for Intel GPUs. Only **Kronos** uses XPU in
this app (TimesFM / Moirai / Chronos-2 fall back to CPU on it). To enable it:

1. **Install the Intel GPU compute runtime** (Ubuntu example):

   ```bash
   sudo apt install intel-opencl-icd libze1
   ```

   Other distros: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/

2. **Switch torch from the CPU build to the XPU build** in `pyproject.toml`:

   ```toml
   [[tool.uv.index]]
   name = "pytorch-xpu"
   url = "https://download.pytorch.org/whl/xpu"
   explicit = true

   [tool.uv.sources]
   torch = { index = "pytorch-xpu" }
   ```

   Then `uv sync --python 3.11 --reinstall-package torch`.

3. **(Optional)** Install Intel Extension for PyTorch for extra optimized ops:

   ```bash
   uv pip install intel-extension-for-pytorch
   ```

4. **Verify** the XPU backend is detected, then launch the app and select `xpu`
   in the sidebar (with a **Kronos** model):

   ```bash
   uv run python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
   uv run streamlit run app.py
   ```

Please note that CPU-only / CUDA torch builds have no XPU support, and Intel
Arc / Iris Xe / Core Ultra iGPUs have the best XPU coverage.

### Intel NPU (AI Boost, Core Ultra) — via OpenVINO

PyTorch has **no native NPU backend**; Intel NPUs are reached through
**OpenVINO**. This app exposes the NPU as an `npu` device in the sidebar and
runs **Kronos** on it (the only backend with an NPU path; TimesFM / Moirai /
Chronos-2 fall back to CPU). Kronos is a custom autoregressive model, so we
compile it with OpenVINO's torch backend rather than exporting to IR:

```bash
uv pip install openvino
uv run streamlit run app.py   # then pick `npu` in the sidebar (Kronos model)
```

When `npu` is selected, `_load_kronos` runs
`torch.compile(model, backend="openvino")`; the NPU has no `torch.device`
string, so the compiled module still runs on CPU tensors while OpenVINO
offloads operators to the NPU. If OpenVINO's torch backend is unavailable the
app logs a warning and falls back to CPU. This is the experimental path;
**Kronos is dynamic-shape and NPU support varies by model/size**. For the most
reliable Intel acceleration today, use the GPU via XPU (above) or an NVIDIA GPU
via CUDA.

## Tests

```bash
uv run pytest
```

## Project layout

```
app.py              # Streamlit UI
src/
  data.py           # yfinance -> normalized OHLCV DataFrame
  models.py         # registry (timesfm + moirai + kronos + chronos2), caching, predictors
  predict.py        # forecast and backtest on top of a predictor
  backtest.py       # metrics: MAE/RMSE/MAPE/directional accuracy
  plotting.py       # Plotly candlestick charts
vendor/Kronos       # vendored Kronos repo (gitignored, clone manually)
tests/              # unit tests (no network, no model)
```

## Notes and limitations

- **No model predicts the market reliably**; backtests are meant to assess
  forecast quality for each specific ticker/timeframe.
- Moirai and TimesFM 3.0 weights are licensed **CC-BY-NC-4.0** / **non-commercial**;
  Kronos and Chronos-2 are Apache-2.0.
- `open/high/low/close/volume` are reconciled so `high`/`low` always enclose the
  `open`/`close` body. TimesFM 3.0, Moirai and Chronos-2 forecast them jointly;
  Kronos samples them as a sequence.
- Future timestamps use a fixed frequency: exact for crypto (24/7),
  approximate for stocks (nights/weekends).
- yfinance is an unofficial API: limited intraday history (e.g. 1m ≈ 7 days,
  1h ≈ 2 years) and possible rate limits.
- On CPU, forecasting returns in seconds-to-tens-of-seconds depending on the
  model, context and `pred_len`.