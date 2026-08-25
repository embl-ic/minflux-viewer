r"""Benchmark the Attribute Plot CPU reduction on real and synthetic data.

Examples
--------
Two real files plus a synthetic selection at 80% of the startup GPU-memory
edge::

    .\.venv\Scripts\python.exe scripts\benchmark_attribute_cpu.py \
        "D:\data\sample.mat" "D:\data\batch.msr" --synthetic-fraction 0.8

The script never writes or modifies source data.  It uses the same loader/raw
iteration accessor as the viewer, aggregates an ``idx``/``efo`` and/or
``idx``/``tim`` view into 900x700 display cells, and emits JSON lines suitable
for keeping as a performance record.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minflux_viewer.core.app_state import default_prefs  # noqa: E402
from minflux_viewer.core.loader import (  # noqa: E402
    load_dataset,
    load_from_mfx_array,
    load_npy,
    mfx_get,
)
from minflux_viewer.ui.attribute_cpu import (  # noqa: E402
    aggregate_screen_points,
    joint_extent,
)

LOAD_PREFS = default_prefs()
LOAD_PREFS["data"].update({
    "estimate_z_scaling_factor": False,
    "compute_loc_prec": False,
    "compute_local_density": False,
})


def _rss_mib() -> float:
    return psutil.Process().memory_info().rss / (1024**2)


def _datasets(path: Path):
    if path.suffix.lower() == ".npy":
        return [(path.stem, load_npy(path, prefs=LOAD_PREFS))]
    if path.suffix.lower() != ".msr":
        return [(path.stem, load_dataset(path, prefs=LOAD_PREFS))]
    from minflux_viewer.msr.msr_parser import GeneralMSRParser

    parsed = GeneralMSRParser().parse(str(path), None, log=lambda *_a, **_k: None)
    result = []
    for name, mfx in (parsed.get("mfx_map") or {}).items():
        array = np.asarray(mfx)
        if array.size:
            result.append((
                str(name),
                load_from_mfx_array(
                    array,
                    name=str(name),
                    folder=str(path.parent),
                    prefs=LOAD_PREFS,
                ),
            ))
    return result


def _aggregate_pair(dataset, y_name: str, width: int, height: int) -> dict:
    started = time.perf_counter()
    x = np.asarray(
        mfx_get(dataset, "idx", itr="all", vld_only=False)
    ).ravel()
    y_raw = mfx_get(dataset, y_name, itr="all", vld_only=False)
    if y_raw is None:
        raise ValueError(f"no {y_name} attribute")
    y = np.asarray(y_raw).ravel()
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    extent = joint_extent(x, y)
    if extent is None:
        return {
            "attribute": y_name,
            "rows": n,
            "drawable": 0,
            "seconds": time.perf_counter() - started,
        }
    aggregation = aggregate_screen_points(
        x,
        y,
        bounds=extent,
        width=width,
        height=height,
    )
    return {
        "attribute": y_name,
        "rows": n,
        "drawable": aggregation.drawable_count,
        "visible": aggregation.visible_count,
        "occupied_cells": aggregation.occupied_count,
        "seconds": time.perf_counter() - started,
    }


def _artifact_monotonicity(dataset, width: int, height: int) -> dict:
    """Reproduce the reported zoom and prove valid is a subset on screen."""

    started = time.perf_counter()
    selections = {}
    for label, valid_only in (("all", False), ("valid", True)):
        x = np.asarray(
            mfx_get(dataset, "idx", itr="all", vld_only=valid_only)
        ).ravel()
        y = np.asarray(
            mfx_get(dataset, "efo", itr="all", vld_only=valid_only)
        ).ravel()
        c = np.asarray(
            mfx_get(dataset, "vld", itr="all", vld_only=valid_only)
        ).ravel()
        n = min(x.size, y.size, c.size)
        selections[label] = (x[:n], y[:n], c[:n])
    extent = joint_extent(*selections["all"][:2])
    if extent is None:
        return {"error": "no drawable idx/efo rows"}
    bounds = (0.0, 30_000.0, extent[2], extent[3])
    grids = {
        label: aggregate_screen_points(
            x, y, values=c, bounds=bounds, width=width, height=height
        )
        for label, (x, y, c) in selections.items()
    }
    all_counts = grids["all"].counts
    valid_counts = grids["valid"].counts
    violating = int(np.count_nonzero(valid_counts > all_counts))
    return {
        "view": "idx 0..30000 vs efo, all iterations",
        "all_visible": grids["all"].visible_count,
        "valid_visible": grids["valid"].visible_count,
        "cells_where_valid_exceeds_all": violating,
        "monotonic": violating == 0,
        "count_plus_mean_c_seconds": time.perf_counter() - started,
    }


def benchmark_file(path: Path, width: int, height: int) -> list[dict]:
    rss_before = _rss_mib()
    load_started = time.perf_counter()
    datasets = _datasets(path)
    load_seconds = time.perf_counter() - load_started
    if not datasets:
        return [{
            "kind": "real",
            "source": str(path),
            "skipped": "no MINFLUX dataset found in container",
            "load_seconds": load_seconds,
            "rss_before_mib": rss_before,
            "rss_after_mib": _rss_mib(),
        }]
    rows = []
    for name, dataset in datasets:
        result = {
            "kind": "real",
            "source": str(path),
            "dataset": name,
            "load_seconds": load_seconds,
            "rss_before_mib": rss_before,
        }
        pairs = []
        for attribute in ("efo", "tim"):
            try:
                pairs.append(_aggregate_pair(dataset, attribute, width, height))
            except Exception as exc:  # keep corpus run moving and report the file
                pairs.append({"attribute": attribute, "error": str(exc)})
        result["pairs"] = pairs
        try:
            result["artifact_check"] = _artifact_monotonicity(
                dataset, width, height
            )
        except Exception as exc:
            result["artifact_check"] = {"error": str(exc)}
        result["rss_after_mib"] = _rss_mib()
        rows.append(result)
    del datasets
    gc.collect()
    return rows


def benchmark_synthetic(fraction: float, width: int, height: int) -> dict:
    from minflux_viewer.ui.gpu_capabilities import point_limit_from_memory

    # This is intentionally a CPU-only benchmark. Starting and destroying a
    # throw-away Qt/OpenGL context in the same process as a ~1 GiB allocation
    # can make Windows driver teardown contaminate the result. The app probes
    # GPU availability separately at startup; for this stress edge, use the
    # same conservative unknown/shared-GPU memory formula directly.
    system_available = int(psutil.virtual_memory().available)
    point_edge = point_limit_from_memory(
        available_system_memory_bytes=system_available,
        free_gpu_memory_bytes=None,
    )
    n = max(1, int(point_edge * max(0.0, min(1.0, fraction))))
    rss_before = _rss_mib()
    allocate_started = time.perf_counter()
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    # Sparse deterministic outliers challenge the spatial extent/reduction and
    # ensure the synthetic view is not only one perfectly straight diagonal.
    step = max(1, n // 10_000)
    y[::step] *= -1.0
    allocate_seconds = time.perf_counter() - allocate_started
    started = time.perf_counter()
    aggregation = aggregate_screen_points(
        x,
        y,
        bounds=(0.0, 1.0, -1.0, 1.0),
        width=width,
        height=height,
    )
    elapsed = time.perf_counter() - started
    return {
        "kind": "synthetic",
        "edge_source": "available system memory (conservative shared/unknown GPU formula)",
        "memory_derived_point_edge": point_edge,
        "available_system_memory_mib": system_available / (1024**2),
        "fraction": fraction,
        "rows": n,
        "input_mib": (x.nbytes + y.nbytes) / (1024**2),
        "allocate_seconds": allocate_seconds,
        "aggregate_seconds": elapsed,
        "drawable": aggregation.drawable_count,
        "occupied_cells": aggregation.occupied_count,
        "rss_before_mib": rss_before,
        "rss_after_mib": _rss_mib(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument(
        "--synthetic-fraction",
        type=float,
        default=0.0,
        help="fraction (0..1) of the startup memory-derived GPU point edge",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.paths:
        try:
            rows.extend(benchmark_file(path, args.width, args.height))
        except Exception as exc:
            rows.append({"kind": "real", "source": str(path), "error": str(exc)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    if args.synthetic_fraction > 0:
        synthetic = benchmark_synthetic(
            args.synthetic_fraction, args.width, args.height
        )
        rows.append(synthetic)
        print(json.dumps(synthetic, ensure_ascii=False), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if not any("error" in row for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
