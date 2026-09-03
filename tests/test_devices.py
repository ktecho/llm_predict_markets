"""Unit tests for compute-device detection and labelling (CPU / CUDA / XPU / NPU / MPS).

These tests are hermetic: they inject a *fake* torch module into ``sys.modules``
so they run without a real torch install or GPU.
"""

import importlib
import sys
import types


def _load_models(device_name, *, cuda_available=True):
    """Builds a fake torch and (re)loads :mod:`src.models` against it."""
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace()
    cuda = types.SimpleNamespace()
    cuda.is_available = lambda: cuda_available
    cuda.get_device_name = lambda i: device_name
    xpu = types.SimpleNamespace()
    xpu.is_available = lambda: False
    torch.cuda = cuda
    torch.xpu = xpu
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    torch.get_num_threads = lambda: 8
    sys.modules["torch"] = torch
    return importlib.reload(__import__("src.models", fromlist=["*"])), torch


def test_device_details_nvidia_cuda():
    m, _ = _load_models(device_name="NVIDIA GeForce RTX 3060")
    assert m.device_details("cuda") == "NVIDIA GPU (CUDA): NVIDIA GeForce RTX 3060"


def test_device_details_cpu_and_unknown():
    m, _ = _load_models(device_name="irrelevant")
    assert m.device_details("cpu").startswith("CPU (")
    assert m.device_details("does-not-exist") == "does-not-exist"


def test_available_devices_lists_cuda_when_gpu_present():
    m, _ = _load_models(device_name="NVIDIA GeForce RTX 3060")
    devices = m.available_devices()
    assert devices[0] == "cpu"
    assert "cuda" in devices


def test_effective_device_keeps_cpu_and_cuda():
    m, _ = _load_models(device_name="NVIDIA GeForce RTX 3060")
    for backend in ("timesfm", "moirai", "kronos", "chronos2"):
        assert m.effective_device(backend, "cpu") == "cpu"
        assert m.effective_device(backend, "cuda") == "cuda"


def test_effective_device_falls_back_to_cpu_for_non_kronos():
    m, _ = _load_models(device_name="NVIDIA GeForce RTX 3060")
    for backend in ("timesfm", "moirai", "chronos2"):
        assert m.effective_device(backend, "xpu") == "cpu"
        assert m.effective_device(backend, "mps") == "cpu"


def test_effective_device_keeps_xpu_mps_for_kronos():
    m, _ = _load_models(device_name="NVIDIA GeForce RTX 3060")
    assert m.effective_device("kronos", "xpu") == "xpu"
    assert m.effective_device("kronos", "mps") == "mps"


# ---------------------------------------------------------------- Intel NPU ---


def test_device_details_npu_without_openvino():
    # openvino is not installed in the test environment: the label still works.
    m, _ = _load_models(device_name="irrelevant")
    assert m.device_details("npu") == "Intel NPU (OpenVINO)"


def test_available_devices_lists_npu_when_openvino_present():
    m, _ = _load_models(device_name="NVIDIA GeForce RTX 3060")
    m._openvino_npu_available = lambda: True
    devices = m.available_devices()
    assert "npu" in devices


def test_available_devices_omits_npu_when_openvino_absent():
    m, _ = _load_models(device_name="NVIDIA GeForce RTX 3060")
    m._openvino_npu_available = lambda: False
    devices = m.available_devices()
    assert "npu" not in devices


def test_effective_device_falls_back_to_cpu_for_non_kronos_on_npu():
    m, _ = _load_models(device_name="NVIDIA GeForce RTX 3060")
    for backend in ("timesfm", "moirai", "chronos2"):
        assert m.effective_device(backend, "npu") == "cpu"


def test_effective_device_keeps_npu_for_kronos():
    m, _ = _load_models(device_name="NVIDIA GeForce RTX 3060")
    assert m.effective_device("kronos", "npu") == "npu"


def test_compile_kronos_for_npu_falls_back_to_original_model():
    # Without an OpenVINO torch backend, the NPU compile degrades to the
    # original (CPU) model instead of raising.
    m, _ = _load_models(device_name="irrelevant")
    sentinel = object()
    assert m._compile_kronos_for_npu(sentinel) is sentinel
