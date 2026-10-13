"""The CRLB numbers on the real sample files, pinned.

tests/test_crlb.py checks the bound against the paper and its reference code,
and checks *where* each input comes from on a few real files. Neither notices
when a change to loading, iteration selection, background pooling or count
conventions quietly moves the numbers a user reads. This does: every MINFLUX
dataset in the sample folder, with the bound, the ideal bound, the number of
localizations and the background coverage per axis.

The values were validated on 2026-10-13 (the CRLB review, schema crlb/v2) and
are what the code computes now. A deliberate method change that moves them
should regenerate this table and say in its commit which values moved and why.

Skipped file by file when the sample data is absent, so CI skips all of it.
Each file is loaded once and only its numbers are kept: one dataset has a
million localizations.
"""

from __future__ import annotations

import copy
import gc
import math
from pathlib import Path

import pytest

SAMPLE = Path(r"D:\Workspace\Microscopes\MINFLUX\sample data")

_ROTOR = ("rotar_project_confocal_overlay/1_C5C3_PEG_surface_1-25k_overview_FLUX_seq_"
          "Tracking_2D_Rot_faster_L75nm_final_final_high_salt_plus_fuel_1.msr")
_NUP62 = "10_Nup62_215TCO_LD655_50nM_2mM_THP_anti-GFP-PAINT_Ab580_125pM.msr"
_PAINT = "4_3D_PAINT_2C_ratiometric_ver2_ALFA-25pM-A655_rb-50pm_A680.msr"

# file, entry (1-based), localizations, σ_xy, σ_z, ideal σ_xy, ideal σ_z,
# background coverage xy, background coverage z. σ in nm; None = no such axis.
EXPECTED = [
    ("1.msr", 1, 13371, 7.774853978, 9.370672229, 2.658049798, 2.882222979, 1.0, 0.9844439459),
    (_NUP62, 1, 19954, 6.138450322, 4.481460525, 2.072145124, 1.818653348, 1.0, 0.9925829408),
    (_NUP62, 2, 252063, 2.467654724, 3.110951352, 1.227904446, 1.135607883,
     0.9720704744, 0.9408877939),
    ("1_sample_A_1-100_seq_3D_ori_exc_5.msr", 1, 58577, 6.10649721, 6.10649721,
     5.995469988, 5.995469988, 1.0, 1.0),
    ("2_3C_measurement.msr", 1, 32498, 2.844073815, None, 0.6436765855, None, 0.9922456767, 0.0),
    ("2_3C_measurement.msr", 2, 40566, 3.756696748, None, 0.7635037438, None, 0.9926292955, 0.0),
    ("2_3C_measurement.msr", 3, 74539, 3.10173553, None, 0.6477037984, None, 0.990998001, 0.0),
    ("3_Bonly_MRED_75pM_MINFLUX_3D.msr", 1, 182792, 3.410590314, 5.531200649, 1.214285861,
     1.106853649, 0.9891625454, 0.9521040308),
    (_PAINT, 1, 30786, 9.361471187, 11.44031484, 1.227904446, 1.209477499,
     0.9933411291, 0.9819398428),
    (_PAINT, 2, 1037149, 9.229706067, 11.49977984, 1.241991764, 1.227943872,
     0.9944279944, 0.9751164008),
    (_ROTOR, 1, 167978, 2.083689198, None, 2.083689198, None, 0.0, 0.0),
]

_RESULTS: dict[str, dict[int, dict]] = {}


def _number(value):
    return float(value) if value is not None and math.isfinite(value) else None


def _results(name: str) -> dict[int, dict]:
    """Every CRLB-capable dataset of one file, as numbers, computed once."""
    if name not in _RESULTS:
        path = SAMPLE / name
        if not path.is_file():
            pytest.skip(f"sample data not available: {path}")
        from minflux_viewer.analysis.localization_precision import (
            crlb_from_inputs,
            crlb_inputs_for_dataset,
        )
        from minflux_viewer.core.app_state import DEFAULT_PREFS
        from minflux_viewer.msr.quick_open import load_viewer_msr

        rows = {}
        for entry, ds in enumerate(load_viewer_msr(path, prefs=copy.deepcopy(DEFAULT_PREFS)), 1):
            inputs = crlb_inputs_for_dataset(ds, prefs=DEFAULT_PREFS)
            if inputs is None:
                continue
            r = crlb_from_inputs(inputs)
            rows[entry] = dict(
                n=r.n_locs, xy=_number(r.primary_sigma), z=_number(r.primary_sigma_z),
                xy_ideal=_number(r.median_sigma_ideal), z_ideal=_number(r.median_sigma_z_ideal),
                cov_xy=r.background_coverage, cov_z=r.background_coverage_z)
        _RESULTS[name] = rows
        gc.collect()
    return _RESULTS[name]


def _same(actual, expected):
    if expected is None:
        return actual is None
    return actual is not None and actual == pytest.approx(expected, rel=1e-7, abs=1e-12)


@pytest.mark.parametrize(
    "name, entry, n, xy, z, xy_ideal, z_ideal, cov_xy, cov_z", EXPECTED,
    ids=[f"{row[0].rsplit('/', 1)[-1][:24]}#{row[1]}" for row in EXPECTED])
def test_crlb_on_the_sample_files_is_unchanged(name, entry, n, xy, z, xy_ideal, z_ideal,
                                               cov_xy, cov_z):
    rows = _results(name)
    assert entry in rows, f"{name} entry {entry} no longer yields a CRLB"
    got = rows[entry]
    want = dict(n=n, xy=xy, z=z, xy_ideal=xy_ideal, z_ideal=z_ideal, cov_xy=cov_xy, cov_z=cov_z)
    moved = {key: (want[key], got[key]) for key in want if not _same(got[key], want[key])}
    assert not moved, f"{name} entry {entry}: (expected, now) {moved}"


def test_every_file_yields_exactly_the_pinned_datasets():
    """A dataset that newly yields a CRLB (or stops) is a change worth seeing."""
    present = [name for name in sorted({row[0] for row in EXPECTED})
               if (SAMPLE / name).is_file()]
    if not present:
        pytest.skip(f"sample data not available: {SAMPLE}")
    for name in present:
        pinned = sorted(row[1] for row in EXPECTED if row[0] == name)
        assert sorted(_results(name)) == pinned, name
