"""CUDA protocol/lifecycle tests run without CuPy; hardware cases are opt-in."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import fftconvolve

from minflux_viewer.analysis import conv_segmentation as cs
from minflux_viewer.core import cuda_fft as cuda
from minflux_viewer.core import cuda_fft_worker as wire


def _settings():
    return cuda.CudaFftSettings(sys.executable)


def test_smooth_fft_lengths_bound_full_convolution_without_excess_padding():
    from scipy.fft import next_fast_len

    for size in (*range(1, 200), 1075, 2098, 1000001):
        assert wire._fast_length(size) == next_fast_len(size, real=True)


def _fake_worker(python, directory, timeout, check):
    check()
    request = wire.read_metadata(directory / "request.json")
    image = np.load(directory / "image.npy", allow_pickle=False)
    kernel = np.load(directory / "kernel.npy", allow_pickle=False)
    output = fftconvolve(image, kernel, mode="same")
    np.save(directory / "output.npy", output, allow_pickle=False)
    provenance = dict(backend="cuda-cupy", dtype="float64", mode="same", boundary="zero", axes=[0, 1],
                      device=request["device"], fft_shape=list(wire.fft_shape(image.shape, kernel.shape)),
                      input_sha256=request["input_sha256"], output_sha256=wire.array_digest(output))
    result = dict(protocol=wire.PROTOCOL, job_id=request["job_id"], status="ok", provenance=provenance)
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return 0


def test_protocol_round_trip_and_no_cuda_import_in_parent(monkeypatch):
    monkeypatch.setattr(cuda, "_run_worker", _fake_worker)
    image, kernel = np.arange(35.).reshape(5, 7), np.arange(6.).reshape(2, 3)
    result, provenance = cuda.cuda_fftconvolve(image, kernel, _settings())
    np.testing.assert_array_equal(result, fftconvolve(image, kernel, mode="same"))
    assert provenance["wall_seconds"] >= 0 and provenance["protocol"] == 1
    assert not any(name == "cupy" or name.startswith("cupy.") for name in sys.modules)


@pytest.mark.parametrize("field,value", [
    ("protocol", 99), ("job_id", "wrong"), ("status", "invalid"), ("provenance", None),
    ("provenance.output_sha256", "wrong"), ("provenance.input_sha256", []),
    ("provenance.dtype", "float32"), ("provenance.axes", [1, 0]),
])
def test_mismatched_results_never_publish(monkeypatch, field, value):
    def broken(*args):
        _fake_worker(*args)
        path = args[1] / "result.json"
        result = wire.read_metadata(path)
        if field.startswith("provenance."):
            result["provenance"][field.split(".")[1]] = value
        else:
            result[field] = value
        path.write_text(json.dumps(result), encoding="utf-8")
        return 0
    monkeypatch.setattr(cuda, "_run_worker", broken)
    with pytest.raises(cuda.CudaFftError):
        cuda.cuda_fftconvolve(np.ones((3, 4)), np.ones((2, 2)), _settings())


def test_crash_failure_releases_admission_and_recovers(monkeypatch):
    monkeypatch.setattr(cuda, "_run_worker", lambda *args: -11)
    with pytest.raises(cuda.CudaFftError, match="without a result"):
        cuda.cuda_fftconvolve(np.ones((3, 4)), np.ones((2, 2)), _settings())
    monkeypatch.setattr(cuda, "_run_worker", _fake_worker)
    cuda.cuda_fftconvolve(np.ones((3, 4)), np.ones((2, 2)), _settings())


def test_busy_device_refuses_without_starting_process(monkeypatch):
    monkeypatch.setattr(cuda, "_run_worker", lambda *args: pytest.fail("started worker"))
    with cuda._DEVICE_LOCK, pytest.raises(cuda.CudaFftError, match="active"):
        cuda.cuda_fftconvolve(np.ones((3, 4)), np.ones((2, 2)), _settings())


def test_failure_and_delivery_cancellation_clean_job_directory(monkeypatch):
    directories = []
    def cancelled(*args):
        directories.append(args[1])
        _fake_worker(*args)
        raise InterruptedError("cancelled")
    monkeypatch.setattr(cuda, "_run_worker", cancelled)
    with pytest.raises(InterruptedError):
        cuda.cuda_fftconvolve(np.ones((3, 4)), np.ones((2, 2)), _settings())
    assert all(not directory.exists() for directory in directories)
    assert cuda._DEVICE_LOCK.acquire(blocking=False)
    cuda._DEVICE_LOCK.release()


def test_external_environment_sanitizes_frozen_native_paths(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", "/frozen/native", raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join(["/frozen/native", "/normal/bin"]))
    monkeypatch.setenv("PYTHONPATH", "/foreign/python")
    monkeypatch.setenv("PYTHONHOME", "/foreign/python")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/frozen/native")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/system/lib")
    env = cuda._worker_environment(Path(sys.executable))
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert "/frozen/native" not in env["PATH"]
    assert env["LD_LIBRARY_PATH"] == "/system/lib"


@pytest.mark.parametrize("image,kernel", [
    (np.empty((0, 2)), np.ones((2, 2))), (np.ones(3), np.ones((2, 2))),
    (np.ones((3, 4)) * np.nan, np.ones((2, 2))),
    (np.ones((3, 4), dtype=complex), np.ones((2, 2))),
])
def test_invalid_arrays_refuse_before_process(monkeypatch, image, kernel):
    monkeypatch.setattr(cuda, "_run_worker", lambda *args: pytest.fail("started worker"))
    with pytest.raises(ValueError):
        cuda.cuda_fftconvolve(image, kernel, _settings())


@pytest.mark.parametrize("kwargs", [
    dict(python_executable=""), dict(python_executable="python"), dict(device=-1),
    dict(device=True), dict(budget_bytes=0), dict(timeout_seconds=float("nan")),
])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        replace(_settings(), **kwargs).validate()


def test_resource_estimate_precedes_conversion_and_spawn(monkeypatch):
    monkeypatch.setattr(cuda, "_run_worker", lambda *args: pytest.fail("started worker"))
    with pytest.raises(ValueError, match="budget"):
        cuda.cuda_fftconvolve(np.ones((3, 4)), np.ones((2, 2)), replace(_settings(), budget_bytes=1))


def test_gpu_preflight_precedes_histogram_and_kernel_allocation(monkeypatch):
    monkeypatch.setattr(cs, "build_kernel", lambda *a, **kw: pytest.fail("allocated kernel"))
    with pytest.raises(ValueError, match="device budget"):
        cs.convolution_maps(np.array([[0., 0.], [10., 10.], [20., 20.]]),
                            model_key="ring", params={}, pixel_size_nm=4.,
                            cuda_settings=replace(_settings(), budget_bytes=1))


@pytest.mark.parametrize("kind", ["cancel", "timeout"])
def test_owned_process_stopped_and_reaped_on_cancel_or_timeout(monkeypatch, tmp_path, kind):
    class FakeProcess:
        killed = False
        reaped = False
        def __init__(self, args, **kwargs):
            assert args[1] == "-I" and kwargs["stdin"] == subprocess.DEVNULL
        def poll(self):
            return -9 if self.killed else None
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.reaped = True
            return -9
        def kill(self):
            self.killed = True
    process = FakeProcess([None, "-I"], stdin=subprocess.DEVNULL)
    monkeypatch.setattr(cuda.subprocess, "Popen", lambda *a, **kw: process)
    calls = []
    def check():
        calls.append(True)
        if kind == "cancel" and len(calls) > 1:
            raise InterruptedError("cancelled")
    with pytest.raises((InterruptedError, cuda.CudaFftError)):
        cuda._run_worker(Path(sys.executable), tmp_path, -1, check)
    assert process.killed and process.reaped


def test_metadata_and_array_header_bounds(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("x" * (wire.MAX_METADATA + 1))
    with pytest.raises(ValueError, match="size limit"):
        wire.read_metadata(path)
    output = tmp_path / "output.npy"
    np.save(output, np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="shape/dtype"):
        wire.load_array(output, (2, 3))
    np.save(output, np.full((2, 3), np.inf))
    with pytest.raises(ValueError, match="nonfinite"):
        wire.load_array(output, (2, 3))


def test_maps_keep_cpu_options_and_detection_parity(monkeypatch):
    monkeypatch.setattr(cuda, "_run_worker", _fake_worker)
    xy = np.random.default_rng(38).normal(size=(300, 2)) * 100
    for dog in (False, True):
        kwargs = dict(model_key="ring", params={}, pixel_size_nm=4., sqrt_stabilize=True, dog=dog)
        cpu = cs.convolution_maps(xy, **kwargs)
        gpu = cs.convolution_maps(xy, **kwargs, cuda_settings=_settings())
        np.testing.assert_array_equal(cpu["response_map"], gpu["response_map"])
        assert gpu["fft_provenance"]["backend"] == "cuda-cupy"


def _hardware_settings():
    python = os.environ.get("MFV_CUDA_PYTHON")
    if not python:
        pytest.skip("set MFV_CUDA_PYTHON to an external CuPy Python for hardware validation")
    return cuda.CudaFftSettings(python)


@pytest.mark.parametrize("shape,kshape", [
    ((9, 13), (3, 5)), ((8, 12), (4, 2)), ((7, 8), (10, 9)), ((1, 9), (2, 3)), ((1, 1), (1, 1)),
])
def test_hardware_fft_parity(shape, kshape):
    settings = _hardware_settings()
    rng = np.random.default_rng(38)
    image, kernel = rng.normal(size=shape), rng.normal(size=kshape)
    # Signed, nonsymmetric kernels catch correlation and even-crop mistakes.
    image[0, 0] += 100
    image[-1, -1] -= 50
    result, provenance = cuda.cuda_fftconvolve(image, kernel, settings)
    np.testing.assert_allclose(result, fftconvolve(image, kernel, mode="same"), atol=1e-11, rtol=1e-11)
    assert provenance["device_name"] and provenance["dtype"] == "float64"


def test_hardware_maps_and_detection_parity():
    settings = _hardware_settings()
    xy = np.random.default_rng(77).normal(size=(3000, 2)) * 300
    for dog in (False, True):
        kwargs = dict(model_key="ring", params={}, pixel_size_nm=4., sqrt_stabilize=True, dog=dog)
        cpu = cs.convolution_maps(xy, **kwargs)
        gpu = cs.convolution_maps(xy, **kwargs, cuda_settings=settings)
        np.testing.assert_allclose(cpu["response_map"], gpu["response_map"], atol=1e-11, rtol=1e-11)
        detect = dict(model_key="ring", params={}, pixel_size_nm=4., min_response=.35,
                      min_distance_nm=100., ring_validation=dict(min_support=.1))
        left = cs.threshold_response_maps(xy, cpu, **detect)
        right = cs.threshold_response_maps(xy, gpu, **detect)
        np.testing.assert_array_equal(left["centers"], right["centers"])
        np.testing.assert_allclose(left["peak_values"], right["peak_values"], atol=1e-11, rtol=1e-11)


def test_hardware_zero_impulse_and_recovery_after_cancel_timeout():
    import time

    settings = _hardware_settings()
    image, kernel = np.zeros((8, 13)), np.arange(12.).reshape(3, 4)
    zero, _ = cuda.cuda_fftconvolve(image, kernel, settings)
    np.testing.assert_array_equal(zero, image)
    started = time.monotonic()
    def stop():
        if time.monotonic() - started >= .1:
            raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        cuda.cuda_fftconvolve(image, kernel, settings, check_cancelled=stop)
    with pytest.raises(cuda.CudaFftError, match="timed out"):
        cuda.cuda_fftconvolve(image, kernel, replace(settings, timeout_seconds=.01))
    image[0, 0] = 1
    image[-1, -1] = 2
    result, _ = cuda.cuda_fftconvolve(image, kernel, settings)
    np.testing.assert_allclose(result, fftconvolve(image, kernel, mode="same"), atol=1e-12, rtol=1e-12)


def test_low_free_vram_refuses_before_any_device_array(monkeypatch, tmp_path):
    from types import SimpleNamespace

    class Device:
        def __init__(self, index):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    fake = SimpleNamespace(cuda=SimpleNamespace(Device=Device, runtime=SimpleNamespace(memGetInfo=lambda: (1, 2))))
    monkeypatch.setitem(sys.modules, "cupy", fake)
    image = np.zeros((2, 3))
    kernel = np.ones((2, 2))
    np.save(tmp_path / "image.npy", image)
    np.save(tmp_path / "kernel.npy", kernel)
    request = dict(protocol=wire.PROTOCOL, operation="fftconvolve2d-same-f64", device=0,
                   budget_bytes=512 * 1024**2, image_shape=image.shape, kernel_shape=kernel.shape,
                   input_sha256=[wire.array_digest(image), wire.array_digest(kernel)])
    with pytest.raises(MemoryError, match="free device memory"):
        wire.execute(tmp_path, request)
