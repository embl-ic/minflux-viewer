r"""Residual-drift-correct a saved HlyB/D cell collection, without re-collecting it.

A saved collection keeps each localization's time and trace id and the path of the
acquisition it came from, so drift can be estimated on the **whole acquisition**
and applied to the cropped cells through their timestamps — the original ROI
geometry is not needed.

Method
------
Drift is estimated by image autocorrelation over time windows, in two 2-D passes:
``(x, y)`` for the lateral drift and ``(x, z)`` for the axial. The estimator's 3-D
path renders a full volume per window (hundreds of MiB here) and is not needed for
a trajectory; the shared ``x`` of the two passes doubles as a consistency check.

The window count is **forced**, not left to the pyMINFLUX heuristic: on these
small, sparse fields that heuristic returns 2-4 windows, which is an offset rather
than a trajectory. Fewer than ``--min-windows`` is refused rather than applied.

MBM fiducial beads are NOT used by default. These acquisitions carry a bead track
(``mbm_map``) that looks like a large uncorrected drift (~250 nm over 10 h), but
the localizations are already stabilized against it online: subtracting it
*disperses* repeated observations of one molecule instead of gathering them.
``--use-beads`` is kept for instruments where that is not so, and the self-check
below catches it either way.

Self-check
----------
The correction is scored on how much the site consolidation merges. Removing real
drift brings a molecule's repeated visits back together, so it must give **fewer
sites and more merged traces**. A correction that does the opposite is reported
and, unless ``--force``, not written. Do not validate with cell-centroid excursion
(dominated by scan order) or with median separation under a distance cut (censored
by the cut).

Example
-------
    .\.venv\Scripts\python.exe scripts\drift_correct_cell_collection.py \
        INPUT.h5 --output CORRECTED.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minflux_viewer.analysis.drift_correction import estimate_drift  # noqa: E402
from minflux_viewer.core.cell_collection import (  # noqa: E402
    CellCollection,
    CellSample,
    load_cell_collection,
    save_cell_collection,
)

LOAD_PREFS = {"data": {"iter_load": "last", "only_valid_locs": True,
                       "estimate_z_scaling_factor": False, "use_fixed_z_scaling_factor": False}}


def acquisition_key(cell: CellSample) -> tuple[str, str]:
    """(source file, acquisition display name) — cells sharing this share drift."""
    name = cell.dataset
    display = name.split("|")[-1].strip() if "|" in name else name
    return (cell.source_path, display)


# --------------------------------------------------------------------------- #
# Sources of a drift trajectory
# --------------------------------------------------------------------------- #
def bead_drift_trajectory(points, *, grid_s: float = 30.0,
                          min_points: int = 200,
                          min_span_fraction: float = 0.5):
    """Stage motion from MBM fiducial beads, as ``(t, dx, dy, dz, n_beads)`` nm.

    See the module note: on an instrument that stabilizes against these beads
    online this is the motion *already compensated*, not a residual to remove.
    """
    pts = np.asarray(points)
    if pts.size == 0 or "xyz" not in (pts.dtype.names or ()):
        return None
    gri = np.asarray(pts["gri"]).ravel()
    xyz = np.asarray(pts["xyz"], dtype=float) * 1e9
    tim = np.asarray(pts["tim"], dtype=float).ravel()
    finite = np.isfinite(tim) & np.all(np.isfinite(xyz), axis=1)
    gri, xyz, tim = gri[finite], xyz[finite], tim[finite]
    if tim.size == 0 or float(tim.max() - tim.min()) <= 0:
        return None
    span = float(tim.max() - tim.min())
    grid = np.arange(tim.min(), tim.max() + grid_s, grid_s)
    tracks = []
    for bead in np.unique(gri):
        sel = gri == bead
        if int(sel.sum()) < int(min_points):
            continue
        t_b, p_b = tim[sel], xyz[sel]
        if float(t_b.max() - t_b.min()) < min_span_fraction * span:
            continue          # a bead that dies early cannot anchor the run
        order = np.argsort(t_b, kind="stable")
        t_b, p_b = t_b[order], p_b[order]
        head = max(20, t_b.size // 50)
        reference = np.median(p_b[:head], axis=0)
        tracks.append(np.column_stack([
            np.interp(grid, t_b, p_b[:, axis] - reference[axis])
            for axis in range(3)]))
    if not tracks:
        return None
    combined = np.median(np.stack(tracks), axis=0)
    return grid, combined[:, 0], combined[:, 1], combined[:, 2], len(tracks)


def autocorrelation_trajectory(xyz_nm, t, tid, *, n_windows: int,
                               resolution_nm: float, min_windows: int):
    """Residual drift by image autocorrelation, as ``(t, dx, dy, dz, note)`` nm.

    Two 2-D passes — ``(x, y)`` then ``(x, z)`` — so no full volume is rendered.
    Raises ``ValueError`` when the run cannot support enough windows to be a
    trajectory rather than a single offset.
    """
    t = np.asarray(t, dtype=float)
    span = float(t.max() - t.min())
    if span <= 0:
        raise ValueError("acquisition has no time span")
    window = max(span / float(n_windows), 30.0)

    lateral = estimate_drift(xyz_nm[:, 0], xyz_nm[:, 1], None, t, tid,
                             resolution_nm=resolution_nm,
                             time_window_s=window, dims=2)
    ti = np.asarray(lateral.ti, dtype=float)
    if ti.size < int(min_windows):
        raise ValueError(
            f"only {ti.size} usable window(s) of {lateral.time_window_s:.0f} s — "
            f"too coarse to be a drift trajectory")
    dx = np.asarray(lateral.dxt, dtype=float)
    dy = np.asarray(lateral.dyt, dtype=float)

    axial = estimate_drift(xyz_nm[:, 0], xyz_nm[:, 2], None, t, tid,
                           resolution_nm=resolution_nm,
                           time_window_s=window, dims=2)
    ta = np.asarray(axial.ti, dtype=float)
    dz = np.interp(ti, ta, np.asarray(axial.dyt, dtype=float))
    # The two passes share x, so their disagreement is a free quality read-out.
    dx_check = np.interp(ti, ta, np.asarray(axial.dxt, dtype=float))
    spread = float(np.max(np.abs(dx - dx_check))) if ti.size else float("nan")
    note = (f"{ti.size} win/{lateral.time_window_s:.0f}s, "
            f"x-pass agreement {spread:.1f} nm")
    return ti, dx, dy, dz, note


# --------------------------------------------------------------------------- #
# Acquisition access
# --------------------------------------------------------------------------- #
def _parse(source_path: str):
    from minflux_viewer.msr.msr_parser import GeneralMSRParser
    return GeneralMSRParser().parse(str(source_path), None,
                                    log=lambda *a, **k: None)


def _pick(mapping: dict, display: str):
    if display in mapping:
        return mapping[display]
    matches = [key for key in mapping if display in key or key in display]
    return mapping[matches[0]] if len(matches) == 1 else None


def load_acquisition(source_path: str, display: str, z_scaling_factor: float):
    """One acquisition's last-valid localizations as ``(xyz_nm, t, tid)``.

    ``z`` comes back already scaled by *z_scaling_factor* — the frame the analysis works in.
    """
    from minflux_viewer.core.loader import load_from_mfx_array, mfx_get

    parsed = _parse(source_path)
    mfx = _pick(parsed.get("mfx_map") or {}, display)
    if mfx is None:
        raise KeyError(f"cannot identify acquisition {display!r} in "
                       f"{Path(source_path).name}")
    ds = load_from_mfx_array(np.asarray(mfx), name=display,
                             folder=str(Path(source_path).parent),
                             prefs=LOAD_PREFS)

    def column(attr):
        value = mfx_get(ds, attr, itr="last", vld_only=True)
        return None if value is None else np.asarray(value, dtype=float).ravel()

    xyz = np.column_stack([column("loc_x"), column("loc_y"),
                           column("loc_z")]) * 1e9
    xyz[:, 2] *= float(z_scaling_factor)
    return xyz, column("tim"), column("tid")


def acquisition_beads(source_path: str, display: str):
    points = _pick(_parse(source_path).get("mbm_map") or {}, display)
    if points is None or np.asarray(points).size == 0:
        return None
    return np.asarray(points)


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def merge_statistic(collection: CellCollection, *, z_scaling_factor: float,
                    merge_nm: float) -> dict:
    """How much the site consolidation merges — the validator for a correction.

    Removing real drift gathers a molecule's repeated visits into one site, so a
    good correction *lowers* ``n_sites`` and *raises* ``n_merged``.
    """
    from minflux_viewer.analysis.hlyb_staged import (
        Staged3DConfig, _pool_cell_sites)

    cfg = Staged3DConfig(z_scaling_factor=float(z_scaling_factor), run_sensitivity=False,
                         run_stratum_profile=False)
    pooled = _pool_cell_sites(collection.as_cells(), cfg, merge_nm=float(merge_nm))
    sites = pooled["sites"]
    counts = np.asarray(sites["n_traces"], dtype=np.int64)
    return {
        "n_sites": int(sites["centers_nm"].shape[0]),
        "n_repeated": int(np.sum(counts > 1)),
        "n_merged": int(np.sum(np.clip(counts - 1, 0, None))),
    }


def verdict(before: dict, after: dict) -> tuple[bool, str]:
    better = (after["n_sites"] < before["n_sites"]
              and after["n_merged"] > before["n_merged"])
    return better, (
        f"sites {before['n_sites']:,} -> {after['n_sites']:,}, "
        f"repeated {before['n_repeated']:,} -> {after['n_repeated']:,}, "
        f"merged {before['n_merged']:,} -> {after['n_merged']:,}  "
        f"[{'improved' if better else 'WORSE'}]")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def compare(args: argparse.Namespace) -> int:
    """Score two collections against each other on the merging statistic.

    Use it to check a collection re-collected from drift-corrected datasets
    against the original: correcting real drift gathers each molecule's repeated
    visits, so the corrected one must show fewer sites and more merged traces.
    """
    left = load_cell_collection(args.input)
    right = load_cell_collection(args.compare)
    print(f"A: {len(left):>3} cell(s)  {args.input}")
    print(f"B: {len(right):>3} cell(s)  {args.compare}")
    if len(left) != len(right):
        print("note: different cell counts — the comparison is still valid, "
              "but the two pools are not the same material.")
    before = merge_statistic(left, z_scaling_factor=args.z_scaling_factor, merge_nm=args.merge_nm)
    after = merge_statistic(right, z_scaling_factor=args.z_scaling_factor, merge_nm=args.merge_nm)
    better, summary = verdict(before, after)
    print()
    print("merging statistic (B relative to A)")
    print(f"   {summary}")
    print()
    print(
        "B is better: its drift correction is removing real positional "
        "error."
        if better else
        "B is NOT better: whatever was applied disperses repeated "
        "observations rather than gathering them. Do not trust a pair "
        "distance measured from it.")
    return 0 if better else 1


def run(args: argparse.Namespace) -> int:
    collection = load_cell_collection(args.input)
    print(f"loaded {len(collection)} cell(s) from {args.input}")

    groups: dict[tuple[str, str], list[int]] = {}
    for index, cell in enumerate(collection):
        groups.setdefault(acquisition_key(cell), []).append(index)
    print(f"{len(groups)} distinct acquisition(s)\n")

    cells = list(collection.cells)
    failures: list[str] = []
    corrected = 0
    for (source_path, display), members in sorted(groups.items()):
        label = f"{Path(source_path).name if source_path else '?'} :: {display}"
        if not source_path or not Path(source_path).exists():
            failures.append(f"{label}: source file not available")
            continue
        try:
            if args.use_beads:
                beads = acquisition_beads(source_path, display)
                track = bead_drift_trajectory(beads) if beads is not None else None
                if track is None:
                    raise ValueError("no bead covers enough of the acquisition")
                ti, dx, dy, dz, n_beads = track
                note = f"MBM {n_beads} bead(s)"
            else:
                xyz, t_all, tid_all = load_acquisition(source_path, display,
                                                       args.z_scaling_factor)
                ti, dx, dy, dz, note = autocorrelation_trajectory(
                    xyz, t_all, tid_all, n_windows=args.windows,
                    resolution_nm=args.resolution_nm,
                    min_windows=args.min_windows)
        except Exception as exc:                                  # noqa: BLE001
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            continue

        corrected += 1
        print(f"  {label}\n      {note}; excursion "
              f"x {np.ptp(dx):.1f} / y {np.ptp(dy):.1f} / z {np.ptp(dz):.1f} nm",
              flush=True)
        for index in members:
            cell = collection.cells[index]
            t = np.asarray(cell.tim, dtype=float)
            loc = cell.loc_m.copy()
            loc[:, 0] -= np.interp(t, ti, dx) * 1e-9
            loc[:, 1] -= np.interp(t, ti, dy) * 1e-9
            # z was corrected in the Z-scaled frame; the collection stores raw
            # metres and the analysis applies the scale itself.
            loc[:, 2] -= (np.interp(t, ti, dz) / max(args.z_scaling_factor, 1e-9)) * 1e-9
            cells[index] = CellSample(loc, cell.tid, cell.tim, cell.dataset,
                                      cell.roi, cell.source_path)

    result = CellCollection(cells)
    print(f"\n{corrected}/{len(groups)} acquisition(s) corrected")
    if failures:
        print(f"{len(failures)} left UNCORRECTED:")
        for line in failures:
            print(f"   - {line}")

    print("\nself-check (a real correction gathers a molecule's revisits: "
          "fewer sites, more merged)")
    before = merge_statistic(collection, z_scaling_factor=args.z_scaling_factor, merge_nm=args.merge_nm)
    after = merge_statistic(result, z_scaling_factor=args.z_scaling_factor, merge_nm=args.merge_nm)
    better, summary = verdict(before, after)
    print(f"   {summary}")

    if not better and not args.force:
        print("\nREFUSING to write: the correction disperses repeated "
              "observations rather than gathering them, so it is not removing "
              "real drift. Re-run with --force only if you know why.")
        return 1
    if args.output:
        save_cell_collection(args.output, result)
        print(f"\nwrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path,
                        help="write the corrected collection here")
    parser.add_argument("--z-scaling-factor", type=float, default=0.67,
                        help="Z scaling factor used by the analysis (default 0.67)")
    parser.add_argument("--windows", type=int, default=40,
                        help="drift windows per acquisition (default 40)")
    parser.add_argument("--min-windows", type=int, default=8,
                        help="refuse an acquisition that cannot support this "
                             "many windows (default 8)")
    parser.add_argument("--resolution-nm", type=float, default=10.0)
    parser.add_argument("--merge-nm", type=float, default=4.0,
                        help="same-site diameter used by the self-check")
    parser.add_argument("--use-beads", action="store_true",
                        help="take the drift from the MBM fiducials instead — "
                             "wrong on an instrument that already stabilizes "
                             "against them (see the module docstring)")
    parser.add_argument("--compare", type=Path,
                        help="score another collection against this one on the "
                             "merging statistic and exit (no correction is "
                             "estimated) — the gate for a collection you "
                             "re-collected from drift-corrected datasets")
    parser.add_argument("--force", action="store_true",
                        help="write even if the self-check says the correction "
                             "made things worse")
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(compare(parsed) if parsed.compare else run(parsed))
