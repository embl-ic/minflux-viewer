"""Standalone CUDA worker and shared wire contract; no viewer or Qt imports.

Invoked with an external Python in isolated mode. CuPy is imported only inside
execute(), never into the viewer process. This is trusted local code, not a sandbox.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

# Reset before loading NumPy as well as CuPy, but never change the viewer's DLL
# search directory when this module is imported for its protocol helpers.
if __name__ == "__main__" and sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetDllDirectoryW(None)

import numpy as np  # noqa: E402 - standalone DLL bootstrap must precede native imports

PROTOCOL = 1
MAX_METADATA = 16384
DEFAULT_DEVICE_BUDGET = 512 * 1024**2


def _fast_length(size: int) -> int:
    """Smallest 2/3/5-smooth length >= size, without a SciPy worker dependency."""
    best = 1 << (size - 1).bit_length()
    power5 = 1
    while power5 < best:
        power35 = power5
        while power35 < best:
            ratio = (size + power35 - 1) // power35
            candidate = power35 * (1 << (ratio - 1).bit_length())
            best = min(best, candidate)
            power35 *= 3
        power5 *= 5
    return best


def fft_shape(image_shape, kernel_shape) -> tuple[int, int]:
    """Deterministic smooth zero padding; avoids nearly 4x power-of-two waste."""
    if len(image_shape) != 2 or len(kernel_shape) != 2:
        raise ValueError("CUDA convolution requires two-dimensional inputs")
    dims = [int(a) + int(b) - 1 for a, b in zip(image_shape, kernel_shape)]
    if any(int(v) < 1 for v in (*image_shape, *kernel_shape)):
        raise ValueError("CUDA convolution requires nonempty inputs")
    return _fast_length(dims[0]), _fast_length(dims[1])


def device_estimate(image_shape, kernel_shape) -> int:
    # Inputs/output, padded real/complex buffers, FFT workspace plus native
    # headroom. cuFFT/context allocations outside the pool are not a hard quota.
    shape = fft_shape(image_shape, kernel_shape)
    return (128 * math.prod(shape) + 8 * (2 * math.prod(image_shape)
            + math.prod(kernel_shape)) + 64 * 1024**2)


def array_digest(array: np.ndarray) -> str:
    data = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((data.shape, data.dtype.str)).encode("ascii"))
    digest.update(data.data.cast("B"))
    return digest.hexdigest()


def read_metadata(path: Path) -> dict:
    if path.stat().st_size > MAX_METADATA:
        raise ValueError("CUDA worker metadata exceeds size limit")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("CUDA worker metadata must be an object")
    return value


def load_array(path: Path, shape) -> np.ndarray:
    expected = tuple(int(v) for v in shape)
    if path.stat().st_size > 8 * math.prod(expected) + 4096:
        raise ValueError("CUDA array exceeds declared size")
    # Check the header before np.load can allocate based on an untrusted shape.
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError("CUDA array must use NPY 1.0")
        actual, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        if actual != expected or dtype != np.dtype("float64"):
            raise ValueError("CUDA array shape/dtype mismatch")
        stream.seek(0)
        array = np.load(stream, allow_pickle=False)
    if not np.isfinite(array).all():
        raise ValueError("CUDA array contains nonfinite values")
    return array


def execute(directory: Path, request: dict) -> dict:
    import cupy as cp

    if request.get("protocol") != PROTOCOL or request.get("operation") != "fftconvolve2d-same-f64":
        raise ValueError("Unsupported CUDA worker protocol/operation")
    device, budget = request["device"], request["budget_bytes"]
    if type(device) is not int or device < 0 or type(budget) is not int or budget < 1:
        raise ValueError("Invalid CUDA device/budget")
    required = device_estimate(request["image_shape"], request["kernel_shape"])
    if required > budget:
        raise ValueError("CUDA convolution estimate exceeds device budget")
    image = load_array(directory / "image.npy", request["image_shape"])
    kernel = load_array(directory / "kernel.npy", request["kernel_shape"])
    if [array_digest(image), array_digest(kernel)] != request["input_sha256"]:
        raise ValueError("CUDA input fingerprint mismatch")
    with cp.cuda.Device(device):
        free, total = cp.cuda.runtime.memGetInfo()
        if required > int(free * .8):
            raise MemoryError(f"CUDA estimate {required} bytes exceeds 80% of free device memory ({free})")
        cp.get_default_memory_pool().set_limit(size=budget)
        cp.fft.config.get_plan_cache().set_size(0)
        cp.fft.config.get_plan_cache().set_memsize(0)
        shape = fft_shape(image.shape, kernel.shape)
        start = time.perf_counter()
        left = cp.fft.rfftn(cp.asarray(image), s=shape, axes=(0, 1))
        right = cp.fft.rfftn(cp.asarray(kernel), s=shape, axes=(0, 1))
        left *= right
        del right
        full = cp.fft.irfftn(left, s=shape, axes=(0, 1))
        # SciPy's same output is centered with respect to the full convolution,
        # with the first input's shape; even kernels must not shift by one pixel.
        row, col = ((n - 1) // 2 for n in kernel.shape)
        output = cp.asnumpy(full[row:row + image.shape[0], col:col + image.shape[1]])
        elapsed = time.perf_counter() - start  # asnumpy synchronizes this work
        if not np.isfinite(output).all():
            raise ValueError("CUDA produced nonfinite output")
        props = cp.cuda.runtime.getDeviceProperties(device)
        name = props["name"]
        provenance = dict(backend="cuda-cupy", cupy=cp.__version__, numpy=np.__version__,
                          device=device, device_name=name.decode() if isinstance(name, bytes) else str(name),
                          compute_capability=cp.cuda.Device(device).compute_capability,
                          driver=cp.cuda.runtime.driverGetVersion(), runtime=cp.cuda.runtime.runtimeGetVersion(),
                          dtype="float64", axes=[0, 1], mode="same", boundary="zero",
                          fft_shape=list(shape), device_estimate_bytes=required,
                          device_total_bytes=total, device_free_bytes=free, compute_seconds=elapsed,
                          input_sha256=request["input_sha256"], output_sha256=array_digest(output))
        np.save(directory / "output.npy", output, allow_pickle=False)
        return provenance


def main() -> int:
    directory = Path(sys.argv[1]).resolve(strict=True)
    request = read_metadata(directory / "request.json")
    result = dict(protocol=PROTOCOL, job_id=request.get("job_id"))
    try:
        result.update(status="ok", provenance=execute(directory, request))
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError, MemoryError) as exc:
        # Explicit terminal failure, not substitute output; native crashes are
        # detected by the parent exit code/missing result contract.
        result.update(status="error", error=f"{type(exc).__name__}: {exc}"[:4000])
    encoded = json.dumps(result, allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_METADATA:
        raise ValueError("CUDA result metadata exceeds size limit")
    (directory / "result.json").write_text(encoded, encoding="utf-8")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
