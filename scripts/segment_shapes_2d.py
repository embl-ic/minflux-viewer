r"""Detect and segment objects of a known 2-D geometry (e.g. E. coli cells).

Accepts either a calibrated image or a localization dataset, and writes an
overlay, a label image, a per-object table and a JSON report.

Examples
--------
Rod-shaped cells from MINFLUX localizations (XY, equivalent to a sum-Z
projection).  For a ``.msr`` the acquisition ROI is read from the file and used
as the true field of view, so cells cut by the frame are reported as clipped::

    .\.venv\Scripts\python.exe scripts\segment_shapes_2d.py DATA.msr \
        --output-dir output\cells --length 1400,4000 --width 600,1200

A whole folder, and the bent-rod model for visibly curved cells::

    .\.venv\Scripts\python.exe scripts\segment_shapes_2d.py DATA_DIR \
        --output-dir output\cells --model arc_capsule

A rendered TIFF instead (calibration read from the file, or ``--pixel-size-nm``)::

    .\.venv\Scripts\python.exe scripts\segment_shapes_2d.py RENDER.tif \
        --output-dir output\cells
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minflux_viewer.analysis.shape_segmentation import (  # noqa: E402
    SHAPE_MODELS,
    ShapePrior,
    ShapeSegmentationConfig,
    get_shape_model,
    instance_outline,
    segment_shapes_in_image,
    segment_shapes_in_points,
)

IMAGE_EXTS = {".tif", ".tiff"}
POINT_EXTS = {".msr", ".mat", ".npy", ".csv", ".tsv", ".txt", ".json"}
LOAD_PREFS = {"data": {"iter_load": "last", "only_valid_locs": True,
                       "estimate_z_scaling_factor": False, "use_fixed_z_scaling_factor": False}}
OVERLAY_COLORS = np.array([
    (255, 70, 70), (70, 220, 100), (80, 150, 255), (255, 195, 55),
    (215, 90, 255), (60, 235, 235), (255, 130, 60), (170, 255, 90),
], dtype=np.uint8)


def _range(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected 'low,high', got {text!r}")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if hi < lo:
        raise argparse.ArgumentTypeError(f"high {hi} is below low {lo}")
    return lo, hi


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _datasets_from_msr(path: Path):
    """Every MINFLUX acquisition in a ``.msr``, as ``(name, dataset, did)``."""
    from minflux_viewer.core.loader import load_from_mfx_array
    from minflux_viewer.msr.msr_parser import GeneralMSRParser

    parsed = GeneralMSRParser().parse(str(path), None, log=lambda *a, **k: None)
    dids = {entry.get("display_name"): entry.get("did")
            for entry in (parsed.get("datasets") or [])}
    out = []
    for name, mfx in (parsed.get("mfx_map") or {}).items():
        if mfx is None or np.asarray(mfx).size == 0:
            continue
        out.append((name, load_from_mfx_array(np.asarray(mfx), name=name,
                                              folder=str(path.parent),
                                              prefs=LOAD_PREFS),
                    dids.get(name)))
    return out


def _datasets_from_file(path: Path):
    from minflux_viewer.core.particle_extract import load_any_dataset
    return [(path.stem, load_any_dataset(path, prefs=LOAD_PREFS), None)]


def _xy_nm(dataset) -> tuple[np.ndarray, np.ndarray]:
    from minflux_viewer.core.loader import mfx_get
    x = np.asarray(mfx_get(dataset, "loc_x", itr="last", vld_only=True), float) * 1e9
    y = np.asarray(mfx_get(dataset, "loc_y", itr="last", vld_only=True), float) * 1e9
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def _acquisition_bounds_by_did(path: Path) -> dict[str, tuple]:
    """Per-acquisition field of view of a ``.msr``, keyed by dataset did.

    A ``.msr`` can hold several acquisitions, each scanned over its own ROI, so
    the ROIs must be matched to their dataset by did — unioning every ROI in the
    file yields a box spanning unrelated acquisitions and defeats the whole
    point of knowing the frame.
    """
    try:
        from minflux_viewer.msr.acquisition_roi import (
            group_by_dataset, read_acquisition_rois, union_bounds)
        rois = read_acquisition_rois(str(path))
    except Exception:
        return {}
    out = {}
    for did, group in (group_by_dataset(rois) or {}).items():
        if not group:
            continue
        x0, y0, width, height = union_bounds(group)
        if width > 0 and height > 0:
            out[did] = (x0 * 1e9, y0 * 1e9,
                        (x0 + width) * 1e9, (y0 + height) * 1e9)
    return out


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def _overlay_rgb(density: np.ndarray, result) -> np.ndarray:
    from scipy import ndimage as ndi

    values = np.asarray(density, dtype=float)
    finite = values[np.isfinite(values)]
    hi = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    grey = (255.0 * np.clip(values / hi, 0.0, 1.0)).astype(np.uint8) if hi > 0 \
        else np.zeros(values.shape, dtype=np.uint8)
    rgb = np.repeat(grey[..., None], 3, axis=2)
    for index, mask in enumerate(result.masks):
        edge = mask & ~ndi.binary_erosion(mask, iterations=2)
        rgb[edge] = OVERLAY_COLORS[index % len(OVERLAY_COLORS)]
    return rgb


def _write_outputs(out_dir: Path, stem: str, density: np.ndarray, result,
                   pixel_nm: float, payload: dict, *, write_points: bool,
                   point_labels: np.ndarray | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    overlay = out_dir / f"{stem}_overlay.tif"
    tifffile.imwrite(overlay, _overlay_rgb(density, result), photometric="rgb")
    paths["overlay"] = str(overlay)

    labels = out_dir / f"{stem}_labels.tif"
    tifffile.imwrite(
        labels, result.labels.astype(np.uint16), ome=True,
        metadata={"axes": "YX", "PhysicalSizeX": pixel_nm,
                  "PhysicalSizeXUnit": "nm", "PhysicalSizeY": pixel_nm,
                  "PhysicalSizeYUnit": "nm"})
    paths["labels"] = str(labels)

    table = out_dir / f"{stem}_objects.csv"
    rows = [{"id": i, **item.as_dict()}
            for i, item in enumerate(result.instances, 1)]
    if rows:
        with table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        table.write_text("id\n", encoding="utf-8")
    paths["objects_csv"] = str(table)

    outlines = out_dir / f"{stem}_outlines.csv"
    with outlines.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "vertex", "x_nm", "y_nm"])
        for index, item in enumerate(result.instances, 1):
            for vertex, (vx, vy) in enumerate(instance_outline(item)):
                writer.writerow([index, vertex, f"{vx:.2f}", f"{vy:.2f}"])
    paths["outlines_csv"] = str(outlines)

    if write_points and point_labels is not None:
        points = out_dir / f"{stem}_point_labels.csv"
        with points.open("w", newline="", encoding="utf-8") as handle:
            handle.write("object_id\n")
            np.savetxt(handle, point_labels, fmt="%d")
        paths["point_labels_csv"] = str(points)

    report = out_dir / f"{stem}_report.json"
    payload["outputs"] = paths
    report.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    paths["report"] = str(report)
    return paths


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _prior_from_args(args: argparse.Namespace) -> ShapePrior:
    model = get_shape_model(args.model)
    names = model.size_names
    if names == ("length_nm", "width_nm"):
        lo = (args.length[0], args.width[0])
        hi = (args.length[1], args.width[1])
    elif names == ("length_nm", "width_nm", "bend_deg"):
        lo = (args.length[0], args.width[0], args.bend[0])
        hi = (args.length[1], args.width[1], args.bend[1])
    elif names == ("major_nm", "minor_nm"):
        lo = (args.length[0], args.width[0])
        hi = (args.length[1], args.width[1])
    elif names == ("diameter_nm",):
        lo, hi = (args.width[0],), (args.width[1],)
    else:  # pragma: no cover - guards a newly registered model
        raise SystemExit(f"no CLI mapping for model {args.model!r} "
                         f"with parameters {names}")
    prior = ShapePrior(args.model, lo, hi)
    prior.validate()
    return prior


def _run_one(name: str, density_source, pixel_nm, args, prior, cfg,
             out_dir: Path, stem: str, *, points=None, bounds=None) -> dict | None:
    started = time.time()
    if points is not None:
        x, y = points
        if x.size < args.min_points:
            print(f"  [skip] {name}: {x.size} localizations "
                  f"(below --min-points {args.min_points})")
            return None
        result = segment_shapes_in_points(x, y, prior=prior, cfg=cfg,
                                          bounds_nm=bounds)
        density = result.detection_field
        pixel_nm = result.detection_pixel_nm
    else:
        result = segment_shapes_in_image(density_source, pixel_nm, prior=prior,
                                         cfg=cfg)
        density = density_source
    elapsed = time.time() - started

    print(f"  {name}: {len(result)} object(s), "
          f"{result.stats['n_clipped']} clipped  ({elapsed:.1f}s)")
    for component in result.stats["components"]:
        print(f"     component {component['component_id']}: "
              f"{component['area_nm2']/1e6:.2f} um2 -> k={component['chosen_k']} "
              f"{component['by_k']}")
    for index, item in enumerate(result.instances, 1):
        sizes = "  ".join(f"{key}={value:.0f}"
                          for key, value in item.size().items())
        print(f"     #{index}: centre=({item.center_nm[0]:.0f}, "
              f"{item.center_nm[1]:.0f}) angle={item.angle_deg:.1f}  {sizes}"
              f"  iou={item.iou:.3f} visible={item.visible_fraction:.2f}"
              f"{'  CLIPPED' if item.clipped else ''}")

    payload = {
        "source": str(args.input), "dataset": name, "model": args.model,
        "prior": {"size_lo": list(prior.size_lo), "size_hi": list(prior.size_hi)},
        "config": {"detection_pixel_nm": cfg.detection_pixel_nm,
                   "instance_cost": cfg.instance_cost,
                   "smoothing_nm": cfg.smoothing_nm},
        "field_bounds_nm": list(bounds) if bounds else None,
        "seconds": round(elapsed, 2), "stats": result.stats,
        "objects": [item.as_dict() for item in result.instances],
    }
    _write_outputs(out_dir, stem, density, result, pixel_nm, payload,
                   write_points=args.write_point_labels,
                   point_labels=result.point_labels)
    return payload


def run(args: argparse.Namespace) -> list[dict]:
    prior = _prior_from_args(args)
    cfg = ShapeSegmentationConfig(
        detection_pixel_nm=args.detection_pixel_nm,
        smoothing_nm=args.smoothing_nm,
        instance_cost=args.instance_cost,
        max_instances_per_component=args.max_per_component)
    out_dir = args.output_dir.resolve()

    root = args.input
    if root.is_dir():
        files = sorted(p for p in root.rglob("*")
                       if p.suffix.lower() in (IMAGE_EXTS | POINT_EXTS))
    else:
        files = [root]
    if not files:
        raise SystemExit(f"no supported files found under {root}")

    payloads: list[dict] = []
    for path in files:
        print(f"\n== {path}")
        suffix = path.suffix.lower()
        try:
            if suffix in IMAGE_EXTS:
                from minflux_viewer.core.tiff_source import TiffImageSource
                source = TiffImageSource(path)
                try:
                    series = int(getattr(source.metadata, "series_count", 1) or 1)
                    if series > 1:
                        print(f"  [note] {path.name} holds {series} series; "
                              f"only the first is segmented. If this is a Z "
                              f"stack, project it first and pass the projection")
                    depth = tuple(int(source.axis_size(axis) or 1)
                                  for axis in ("z", "t", "c"))
                    if max(depth) > 1:
                        raise ValueError(
                            f"this is a stack (z, t, c = {depth}); segmentation "
                            f"needs one 2-D plane. Project it first (e.g. the "
                            f"sum-Z projection) and pass that, or segment the "
                            f"localizations directly")
                    image = np.asarray(source.read_plane())
                    px_x = float(args.pixel_size_nm
                                 or source.metadata.pixel_size_x.nm or 0.0)
                    px_y = float(args.pixel_size_nm
                                 or source.metadata.pixel_size_y.nm or 0.0)
                finally:
                    source.close()
                if image.ndim != 2:
                    raise ValueError(
                        f"expected a 2-D scalar plane, got shape {image.shape}")
                if px_x <= 0 or px_y <= 0:
                    raise ValueError(
                        "the image is uncalibrated; pass --pixel-size-nm")
                if not np.isclose(px_x, px_y, rtol=0.01):
                    raise ValueError(
                        f"anisotropic XY pixels ({px_x:g} x {px_y:g} nm) "
                        f"are not supported")
                payload = _run_one(path.name, image, px_x, args, prior, cfg,
                                   out_dir, path.stem)
                if payload:
                    payloads.append(payload)
                continue

            entries = (_datasets_from_msr(path) if suffix == ".msr"
                       else _datasets_from_file(path))
            by_did = (_acquisition_bounds_by_did(path)
                      if suffix == ".msr" and args.use_acquisition_roi else {})
            for name, dataset, did in entries:
                bounds = by_did.get(did)
                if suffix == ".msr" and args.use_acquisition_roi and bounds is None:
                    print(f"  [note] {name}: no acquisition ROI matched; the "
                          f"field falls back to the data extent, so 'clipped' "
                          f"cannot be determined")
                x, y = _xy_nm(dataset)
                stem = f"{path.stem}__{name}" if len(entries) > 1 else path.stem
                payload = _run_one(name, None, None, args, prior, cfg, out_dir,
                                   stem, points=(x, y), bounds=bounds)
                if payload:
                    payloads.append(payload)
        except Exception as exc:  # keep a batch going, but never silently
            print(f"  [error] {path.name}: {type(exc).__name__}: {exc}")
            payloads.append({"source": str(path),
                             "error": f"{type(exc).__name__}: {exc}"})

    summary = out_dir / "summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payloads, indent=2, default=float),
                       encoding="utf-8")
    total = sum(len(p.get("objects", [])) for p in payloads)
    failed = sum(1 for p in payloads if "error" in p)
    print(f"\n{total} object(s) across {len(payloads) - failed} dataset(s); "
          f"{failed} failed. Summary: {summary}")
    return payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path,
                        help="image, localization file, or a folder of them")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="capsule", choices=sorted(SHAPE_MODELS),
                        help="known geometry to fit (default: capsule)")
    parser.add_argument("--length", type=_range, default=(1400.0, 4000.0),
                        metavar="LO,HI",
                        help="object length range in nm (default: 1400,4000)")
    parser.add_argument("--width", type=_range, default=(600.0, 1200.0),
                        metavar="LO,HI",
                        help="object width range in nm; for 'disk' this is the "
                             "diameter (default: 600,1200)")
    parser.add_argument("--bend", type=_range, default=(-80.0, 80.0),
                        metavar="LO,HI",
                        help="arc_capsule only: spine turn in degrees "
                             "(default: -80,80)")
    parser.add_argument("--detection-pixel-nm", type=float, default=20.0)
    parser.add_argument("--smoothing-nm", type=float, default=None,
                        help="foreground smoothing; default is width/6")
    parser.add_argument("--instance-cost", type=float, default=0.25,
                        help="price of one extra object, in nominal object "
                             "areas; raise if touching objects are over-split")
    parser.add_argument("--max-per-component", type=int, default=6)
    parser.add_argument("--min-points", type=int, default=500,
                        help="skip localization datasets smaller than this")
    parser.add_argument("--pixel-size-nm", type=float,
                        help="override image calibration")
    parser.add_argument("--no-acquisition-roi", dest="use_acquisition_roi",
                        action="store_false",
                        help="ignore the .msr acquisition ROI and let the data "
                             "extent define the field (disables clipping "
                             "detection)")
    parser.add_argument("--write-point-labels", action="store_true",
                        help="also write the per-localization object id column")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
