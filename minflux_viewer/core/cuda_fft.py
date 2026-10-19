"""Optional FFT convolution in an owned external CUDA Python process."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np

from minflux_viewer.core import cuda_fft_worker as wire

_DEVICE_LOCK = Lock()  # one external CUDA job at a time per viewer process


class CudaFftError(RuntimeError):
    """Explicit CUDA failure; callers must not silently select a CPU algorithm."""


@dataclass(frozen=True)
class CudaFftSettings:
    python_executable: str
    device: int = 0
    budget_bytes: int = wire.DEFAULT_DEVICE_BUDGET
    timeout_seconds: float = 120.0

    def validate(self) -> Path:
        path = Path(self.python_executable).expanduser()
        if not self.python_executable or not path.is_absolute() or not path.is_file():
            raise ValueError("Select an absolute path to the CUDA environment's Python executable")
        if type(self.device) is not int or self.device < 0:
            raise ValueError("CUDA device index must be a nonnegative integer")
        if type(self.budget_bytes) is not int or self.budget_bytes < 1:
            raise ValueError("CUDA memory budget must be positive")
        if not np.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 3600:
            raise ValueError("CUDA timeout must be in (0, 3600] seconds")
        if getattr(sys, "frozen", False) and path.resolve() == Path(sys.executable).resolve():
            raise ValueError("CUDA requires an external Python environment, not the viewer executable")
        return path

    def preflight(self, image_shape, kernel_shape) -> int:
        self.validate()
        required = self.estimate(image_shape, kernel_shape)
        if required > self.budget_bytes:
            raise ValueError(f"CUDA estimate {required / 1024**2:.1f} MiB exceeds the device budget")
        return required

    @staticmethod
    def estimate(image_shape, kernel_shape) -> int:
        return wire.device_estimate(image_shape, kernel_shape)


def _worker_environment(python: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"):
        env.pop(key, None)
    if "LD_LIBRARY_PATH_ORIG" in env:
        env["LD_LIBRARY_PATH"] = env.pop("LD_LIBRARY_PATH_ORIG")
    elif getattr(sys, "frozen", False):
        env.pop("LD_LIBRARY_PATH", None)
    # Do not inherit a frozen bundle's native DLL search roots.
    if getattr(sys, "frozen", False):
        bundle = str(getattr(sys, "_MEIPASS", ""))
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)
                                     if p and (not bundle or not p.lower().startswith(bundle.lower())))
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    return env


def _run_worker(python: Path, directory: Path, timeout: float, check: Callable[[], None]) -> int:
    script = Path(__file__).with_name("cuda_fft_worker.py")
    if not script.is_file():
        raise CudaFftError("CUDA worker resource is missing from the application")
    check()
    process = subprocess.Popen([str(python), "-I", str(script), str(directory)],
                               env=_worker_environment(python), cwd=directory,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            check()
            if time.monotonic() >= deadline:
                raise CudaFftError(f"CUDA worker timed out after {timeout:g} seconds")
            try:
                process.wait(timeout=.05)
            except subprocess.TimeoutExpired:
                continue
        check()
        return int(process.returncode)
    finally:
        if process.poll() is None:
            process.kill()  # only our isolated process; never a Qt/Python thread
        process.wait()


def cuda_fftconvolve(image, kernel, settings: CudaFftSettings, *,
                     check_cancelled: Callable[[], None] = lambda: None) -> tuple[np.ndarray, dict]:
    """Finite real 2D/3D -> float64, zero boundary, first-input same shape.

    Startup, transfers and synchronization are included in wall_seconds. This
    intentionally cold worker retains no device arrays/plans after completion.
    Device reservations do not coordinate unrelated applications/processes.
    """
    import json

    python = settings.validate()
    image, kernel = np.asarray(image), np.asarray(kernel)
    if image.dtype.kind not in "biuf" or kernel.dtype.kind not in "biuf":
        raise ValueError("CUDA convolution requires real numeric arrays")
    settings.preflight(image.shape, kernel.shape)
    check_cancelled()
    if not _DEVICE_LOCK.acquire(blocking=False):
        raise CudaFftError("Another CUDA convolution is active; retry when it finishes")
    start = time.perf_counter()
    try:
        image = np.ascontiguousarray(image, dtype=np.float64)
        kernel = np.ascontiguousarray(kernel, dtype=np.float64)
        if not np.isfinite(image).all() or not np.isfinite(kernel).all():
            raise ValueError("CUDA convolution requires finite inputs")
        job_id = uuid.uuid4().hex
        axes = list(range(image.ndim))
        request = dict(protocol=wire.PROTOCOL, operation=f"fftconvolve{image.ndim}d-same-f64", job_id=job_id,
                       device=settings.device, budget_bytes=settings.budget_bytes,
                       image_shape=list(image.shape), kernel_shape=list(kernel.shape),
                       input_sha256=[wire.array_digest(image), wire.array_digest(kernel)])
        with TemporaryDirectory(prefix="mfv-cuda-fft-") as temp:
            directory = Path(temp)
            np.save(directory / "image.npy", image, allow_pickle=False)
            np.save(directory / "kernel.npy", kernel, allow_pickle=False)
            (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
            code = _run_worker(python, directory, settings.timeout_seconds, check_cancelled)
            result_path = directory / "result.json"
            if not result_path.is_file():
                raise CudaFftError(f"CUDA worker exited ({code}) without a result; check the external Python/CUDA installation")
            result = wire.read_metadata(result_path)
            if result.get("protocol") != wire.PROTOCOL or result.get("job_id") != job_id:
                raise CudaFftError("CUDA worker result identity/protocol mismatch")
            if result.get("status") != "ok":
                raise CudaFftError(str(result.get("error", "CUDA worker failed"))[:4000])
            if code != 0:
                raise CudaFftError(f"CUDA worker failed with exit code {code}")
            provenance = result.get("provenance")
            if not isinstance(provenance, dict) or provenance.get("backend") != "cuda-cupy":
                raise CudaFftError("CUDA worker provenance is missing/invalid")
            if (provenance.get("input_sha256") != request["input_sha256"]
                    or provenance.get("dtype") != "float64" or provenance.get("mode") != "same"
                    or provenance.get("boundary") != "zero" or provenance.get("axes") != axes
                    or provenance.get("device") != settings.device
                    or provenance.get("fft_shape") != list(wire.fft_shape(image.shape, kernel.shape))):
                raise CudaFftError("CUDA worker numerical contract mismatch")
            output = wire.load_array(directory / "output.npy", image.shape)
            if wire.array_digest(output) != provenance.get("output_sha256"):
                raise CudaFftError("CUDA output fingerprint mismatch")
            check_cancelled()
            provenance["wall_seconds"] = time.perf_counter() - start
            provenance["protocol"] = wire.PROTOCOL
            return output, provenance
    finally:
        _DEVICE_LOCK.release()
