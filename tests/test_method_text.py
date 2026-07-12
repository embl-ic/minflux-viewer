"""Tests for the Methods-text generator (analysis/method_text.py).

Pure-Python rule registry — no Qt. A minimal fake ``state`` is enough: the
renderers only read ``state.datasets`` (empty here, so dataset names fall back to
each event's ``dataset_name``).
"""

from types import SimpleNamespace

import pytest

from minflux_viewer.analysis import method_text as mt


def _state():
    return SimpleNamespace(datasets=[])


def _ev(message, name="ds1", idx=0):
    return {"message": message, "dataset_idx": idx, "dataset_name": name}


# --- regexes fire on the real log strings -------------------------------------

STDDEV_MSG = ("Localization precision (StdDev per trace): combined (n-weighted) "
              "sigma_r = 5.30 nm, sigma_z = 12.00 nm over 1,234 of 2,000 traces "
              "(>=5 loc/trace, raw z).")
STDDEV_AUTO_MSG = ("Computed localization precision for 'ds1' using StdDev per trace: "
                   "median sigma=(5.3, 5.1, 12) nm.")
CRLB_Z_MSG = ("Localization precision (CRLB, Marin-Ries): median σ_xy = 5.20 nm "
              "(background-limited), 3.10 nm (ideal), σ_z = 12.34 nm (L_z = 100 nm), "
              "L = 50 nm, σ_q = 100 nm, median N = 200 photons.")
CRLB_2D_MSG = ("Localization precision (CRLB, Marin-Ries): median σ_xy = 5.20 nm "
               "(background-limited), 3.10 nm (ideal), L = 50 nm, σ_q = 100 nm, "
               "median N = 200 photons.")
FRC_MSG = ("Localization precision (FRC): resolution = 25.30 nm "
           "(1/7 threshold, per-localization, 50,000 points, pixel 2.00 nm).")


def test_stddev_text_and_citation():
    txt = mt.generate_method_text(_state(), [_ev(STDDEV_MSG)])
    assert "localization precision of 'ds1'" in txt
    assert "5.30 nm" in txt and "12.00 nm" in txt
    # Ostersehlt citation + DOI inline (plain text)
    assert mt.CITE_STDDEV[0] in txt
    assert mt.CITE_STDDEV[1] in txt


def test_stddev_auto_message_matches():
    txt = mt.generate_method_text(_state(), [_ev(STDDEV_AUTO_MSG)])
    assert "median precision was (5.3, 5.1, 12) nm" in txt
    assert mt.CITE_STDDEV[1] in txt


def test_crlb_with_axial():
    txt = mt.generate_method_text(_state(), [_ev(CRLB_Z_MSG)])
    assert "Cramér-Rao" in txt
    assert "5.20 nm" in txt
    assert "axial precision (σ_z) of 12.34 nm" in txt
    assert mt.CITE_CRLB[1] in txt


def test_crlb_without_axial():
    txt = mt.generate_method_text(_state(), [_ev(CRLB_2D_MSG)])
    assert "5.20 nm" in txt
    assert "axial precision" not in txt  # no σ_z clause for 2-D
    assert mt.CITE_CRLB[0] in txt


CRLB_BUDGET_MSG = ("Localization precision (CRLB, Marin-Ries): median σ_xy = 5.20 nm "
                   "(background-limited), 3.10 nm (ideal), L = 50 nm, σ_q = 250 nm, "
                   "median N = 200 photons; measured σ_r = 9.50 nm (StdDev/trace) → "
                   "excess σ_fl = 7.95 nm.")


def test_crlb_precision_budget_excess_cites_simuflux():
    txt = mt.generate_method_text(_state(), [_ev(CRLB_BUDGET_MSG)])
    assert "excess error of σ_fl = 7.95 nm" in txt
    assert "σ_fl² + σ_CRB²" in txt
    assert mt.CITE_SIMUFLUX[0] in txt   # SimuFLUX cited only when the budget is present
    # the plain CRLB line (no budget) must NOT cite SimuFLUX
    assert mt.CITE_SIMUFLUX[0] not in mt.generate_method_text(_state(), [_ev(CRLB_2D_MSG)])


def test_frc_two_citations():
    txt = mt.generate_method_text(_state(), [_ev(FRC_MSG)])
    assert "Fourier ring correlation" in txt
    assert "25.30 nm" in txt
    assert mt.CITE_FRC_BANTERLE[0] in txt
    assert mt.CITE_FRC_NIEUWENHUIZEN[0] in txt
    assert mt.CITE_FRC_BANTERLE[1] in txt
    assert mt.CITE_FRC_NIEUWENHUIZEN[1] in txt


def test_citations_deduped():
    events = [_ev(STDDEV_MSG), _ev(STDDEV_MSG, name="ds2", idx=1)]
    txt = mt.generate_method_text(_state(), events)
    # one citation line despite two stddev events
    assert txt.count(mt.CITE_STDDEV[0]) == 1


def test_html_renders_hyperlinks():
    html = mt.generate_method_html(_state(), [_ev(STDDEV_MSG), _ev(FRC_MSG)])
    assert f'<a href="{mt.CITE_STDDEV[1]}">' in html
    assert f'<a href="{mt.CITE_FRC_BANTERLE[1]}">' in html
    assert f'<a href="{mt.CITE_FRC_NIEUWENHUIZEN[1]}">' in html
    # the visible anchor text is the URL itself (so plain copy keeps it)
    assert f'>{mt.CITE_STDDEV[1]}</a>' in html


def test_unmatched_event_kept_verbatim():
    txt = mt.generate_method_text(_state(), [_ev("Some unmapped operation happened")])
    assert "Some unmapped operation happened" in txt


def test_rimf_anisotropy_note_no_url():
    msg = "RIMF for 'ds1': 0.6375 (auto (estimate anisotropy))"
    txt = mt.generate_method_text(_state(), [_ev(msg)])
    assert "anisotropy of 'ds1' was estimated" in txt
    # custom method → inline note, no DOI
    assert "custom, MINFLUX Data Viewer" in txt


def _fake_ds(state_dict, meta, *, ndim=3, ntraces=0, num_loc=0, name="ds"):
    return SimpleNamespace(
        name=name, state=state_dict, metadata=meta,
        prop=SimpleNamespace(num_dim=ndim, num_traces=ntraces, num_loc=num_loc),
    )


def test_msr_overlay_description():
    import math
    ang = math.radians(1.4)
    c, s = math.cos(ang), math.sin(ang)
    transform = {
        "reference_channel": "ch1", "moving_channel": "ch2",
        "matrix_4x4": [[c, -s, 0, 12.3], [s, c, 0, -4.5], [0, 0, 1, 8.1], [0, 0, 0, 1]],
        "translation_nm": [12.3, -4.5], "z_translation_nm": 8.1,
        "rotation_2x2": [[c, -s], [s, c]],
        "matched_bead_count": 7, "rmse_xy_nm": 3.2, "method": "mbm xy rigid plus z translation",
    }
    ref = _fake_ds(
        {"overlay_id": "g", "overlay_order": 1},
        {"msr_dataset_name": "ch1", "valid_num_loc": 1000, "source_version": "m2410",
         "overlay_reference": "ch1", "overlay_alignment_mode": "mbm info",
         "overlay_bead_excluded": [],
         "overlay_transform": {"reference_channel": "ch1", "moving_channel": "ch1",
                               "matrix_4x4": [[1, 0, 0, 0]], "rotation_2x2": [[1, 0], [0, 1]]}},
        ndim=3, ntraces=200, num_loc=1000, name="ch1")
    mover = _fake_ds(
        {"overlay_id": "g", "overlay_order": 2},
        {"msr_dataset_name": "ch2", "valid_num_loc": 2000, "source_version": "m2410",
         "overlay_transform": transform},
        ndim=3, ntraces=300, num_loc=2000, name="ch2")
    state = SimpleNamespace(datasets=[ref, mover])
    ev = {"message": "MSR overlay loaded from 'file.msr': 2 channel(s) [ch1, ch2]; "
                     "reference channel 'ch1'; channel alignment: mbm info; "
                     "beads: all available beads used.",
          "dataset_idx": 0, "dataset_name": "ch1"}
    txt = mt.generate_method_text(state, [ev])
    assert "overlay was loaded from the .msr file 'file.msr'" in txt
    assert "'ch1' (1,000 valid localizations, 3D, 200 trace(s), m2410)" in txt
    assert "'ch2' (2,000 valid localizations, 3D, 300 trace(s), m2410)" in txt
    assert "Channel 'ch1' served as the alignment reference." in txt
    assert "MBM bead fiducials (all available beads were used)" in txt
    assert "using 7 matched bead(s) (XY RMSE 3.20 nm)" in txt
    assert "translated by +12.30 nm in X, -4.50 nm in Y, and +8.10 nm in Z" in txt
    assert "1.40° counterclockwise rotation in the XY plane" in txt
    assert "transform matrix [[" in txt


def test_empty_events():
    txt = mt.generate_method_text(_state(), [])
    assert "No log events were selected" in txt


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
