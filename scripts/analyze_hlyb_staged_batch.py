"""Batch the staged HlyB 3-D analysis over modern MSR acquisitions.

The script is intentionally a reproducibility/validation helper, not an
alternative data model.  It uses the same final-iteration, valid-only flat
m2410 selection as the viewer and calls ``analyze_hlyb_staged_3d`` unchanged.
Acquisition-history copies are deduplicated by their MSR display name; files in
``test_data_for_agg`` and MAT aggregates are not treated as replicates.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from minflux_viewer.analysis.hlyb_staged import Staged3DConfig, analyze_hlyb_staged_3d
from minflux_viewer.msr.io import parse_msr_general


def _condition(path: Path) -> str:
    name = path.stem
    for label in ("Bonly", "BD+eGFPA", "BD+A", "BD"):
        if label.lower() in name.lower():
            return label
    return "unclassified"


def _last_valid(mfx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    names = set(mfx.dtype.names or ())
    if not {"loc", "tid", "itr"}.issubset(names):
        raise ValueError("MFX array lacks loc/tid/itr")
    itr = np.asarray(mfx["itr"]).squeeze()
    if itr.ndim != 1:
        raise ValueError("Batch helper currently expects flat m2410 MFX records")
    finite_itr = itr[np.isfinite(itr)]
    if finite_itr.size == 0:
        raise ValueError("MFX array has no finite iteration indices")
    mask = itr == np.max(finite_itr)
    if "vld" in names:
        mask &= np.asarray(mfx["vld"], dtype=bool).ravel()
    loc = np.asarray(mfx["loc"][mask], dtype=float)
    tid = np.asarray(mfx["tid"][mask])
    tim = np.asarray(mfx["tim"][mask], dtype=float) if "tim" in names else None
    return loc, tid, tim


def _compact_result(display_name: str, source: Path, result: dict) -> dict:
    summary = result["summary"]
    bootstrap = result.get("bootstrap") or {}
    sensitivity = result.get("sensitivity") or []
    return {
        "acquisition": display_name,
        "condition": _condition(source),
        "source_msr": str(source),
        "n_traces_total": int(result["n_traces_total"]),
        "n_traces_used": int(result["n_traces_used"]),
        "n_sites": int(result["n_sites"]),
        "n_sites_used": int(result["n_sites_used"]),
        "retained_site_fraction": float(result["n_sites_used"] / max(result["n_sites"], 1)),
        "n_components": int(result["n_components"]),
        "n_excluded_sites": int(result["n_excluded_sites"]),
        "band_ratio": float(summary["band_ratio"]),
        "band_p": float(summary["band_p"]),
        "excess_peak_nm": float(summary["peak_nm"]),
        "excess_centroid_nm": float(summary["positive_excess_centroid_nm"]),
        "excess_median_nm": float(summary["positive_excess_median_nm"]),
        "robust_short_range_excess": result.get("robust_short_range_excess"),
        "sensitivity_passes": int(result.get("sensitivity_passes", 0)),
        "sensitivity_valid_variants": int(result.get("sensitivity_valid_variants", 0)),
        "bootstrap": bootstrap,
        "sensitivity": sensitivity,
    }


def run(folder: Path, cfg: Staged3DConfig) -> list[dict]:
    files = [path for path in folder.rglob("*.msr")
             if "test_data_for_agg" not in {part.lower() for part in path.parts}]
    rows = []
    seen: set[str] = set()
    temp_root = Path.cwd() / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hlyb-staged-", dir=temp_root) as tmp:
        for source in sorted(files):
            parsed = parse_msr_general(str(source), tmp, log=lambda *_args: None)
            for entry in parsed.get("datasets", []):
                display = str(entry.get("display_name") or entry.get("did") or source.stem)
                mfx = entry.get("_mfx")
                if display in seen or not isinstance(mfx, np.ndarray) or mfx.size == 0:
                    continue
                seen.add(display)
                try:
                    loc, tid, tim = _last_valid(mfx)
                    result = analyze_hlyb_staged_3d(loc, tid, tim, cfg)
                except Exception as exc:
                    rows.append({
                        "acquisition": display, "condition": _condition(source),
                        "source_msr": str(source), "error": str(exc),
                    })
                    print(f"{display}: ERROR {exc}", flush=True)
                    continue
                row = _compact_result(display, source, result)
                rows.append(row)
                print(
                    f"{display}: {row['n_sites_used']}/{row['n_sites']} sites, "
                    f"{row['n_components']} component(s), ratio {row['band_ratio']:.3f}, "
                    f"p={row['band_p']:.4f}, excess centroid "
                    f"{row['excess_centroid_nm']:.2f} nm",
                    flush=True,
                )
    return sorted(rows, key=lambda row: str(row.get("acquisition", "")))


def aggregate_acquisitions(rows: list[dict], *, rng_seed: int = 0) -> dict:
    """Summarize descriptors with acquisitions—not pair counts—as replicates."""
    valid = [row for row in rows if "error" not in row]
    groups = {
        "all": valid,
        # Descriptive quality sensitivity only, never a replacement for the
        # all-acquisition result.  It exposes whether tiny fields dominate.
        "all_ge_150_retained_sites": [row for row in valid
                                      if int(row.get("n_sites_used", 0)) >= 150],
    }
    for condition in sorted({row["condition"] for row in valid}):
        groups[condition] = [row for row in valid if row["condition"] == condition]
    rng = np.random.default_rng(int(rng_seed) + 4241)
    out = {}
    for name, group in groups.items():
        if not group:
            continue
        centroids = np.asarray([row["excess_centroid_nm"] for row in group], dtype=float)
        ratios = np.asarray([row["band_ratio"] for row in group], dtype=float)
        record = {
            "n_acquisitions": int(len(group)),
            "median_excess_centroid_nm": float(np.median(centroids)),
            "range_excess_centroid_nm": [float(centroids.min()), float(centroids.max())],
            "median_band_ratio": float(np.median(ratios)),
            "range_band_ratio": [float(ratios.min()), float(ratios.max())],
        }
        if len(group) >= 3:
            pick = rng.integers(0, len(group), size=(9999, len(group)))
            record["acquisition_bootstrap_centroid_ci95_nm"] = np.quantile(
                np.median(centroids[pick], axis=1), [0.025, 0.975]).tolist()
            record["acquisition_bootstrap_ratio_ci95"] = np.quantile(
                np.median(ratios[pick], axis=1), [0.025, 0.975]).tolist()
        else:
            record["acquisition_bootstrap_unavailable"] = (
                "fewer than three independent acquisitions")
        out[name] = record
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true",
                        help="39 null / 19 sensitivity / 99 bootstrap replicates")
    args = parser.parse_args()
    cfg = Staged3DConfig(
        null_replicates=39 if args.quick else 99,
        sensitivity_replicates=19 if args.quick else 31,
        bootstrap_replicates=99 if args.quick else 399,
    )
    rows = run(args.folder.resolve(), cfg)
    text = json.dumps({
        "config": vars(cfg), "results": rows,
        "acquisition_level_summary": aggregate_acquisitions(rows, rng_seed=cfg.rng_seed),
    }, indent=2, allow_nan=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
