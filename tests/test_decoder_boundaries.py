"""Expected decoder/convergence failures differ from resource/program failures."""
from types import SimpleNamespace

import numpy as np
import pytest


def fail(error):
    def broken(*args, **kwargs):
        raise error("injected")
    return broken


def test_parser_does_not_reclassify_allocation_failure_as_image_only(monkeypatch):
    from minflux_viewer.msr import mfxdta, msr_parser

    monkeypatch.setattr(mfxdta, "read_obf_mfxdta_stacks", fail(MemoryError))
    with pytest.raises(MemoryError, match="injected"):
        msr_parser.GeneralMSRParser().parse("unused.msr", log=lambda *_: None)


def test_mfxdta_store_allocation_failure_is_not_a_skipped_dataset(monkeypatch):
    from minflux_viewer.msr import mfxdta, msr_parser

    monkeypatch.setattr(mfxdta, "extract_zarr_store", fail(MemoryError))
    with pytest.raises(MemoryError, match="injected"):
        msr_parser.GeneralMSRParser()._mfxdta_dataset_entry(0, "ds", b"", lambda *_: [], {}, {}, lambda *_: None)


def test_mfxdta_array_allocation_failure_is_not_a_skipped_dataset(monkeypatch):
    from minflux_viewer.msr import mfxdta, msr_parser

    class Node:
        attrs = {}
        __getitem__ = fail(MemoryError)

    monkeypatch.setattr(mfxdta, "extract_zarr_store", lambda blob: {})
    monkeypatch.setattr(msr_parser.zarr2, "open", lambda *a, **k: {"mfx": Node()})
    with pytest.raises(MemoryError, match="injected"):
        msr_parser.GeneralMSRParser()._mfxdta_dataset_entry(0, "ds", b"", lambda *_: [], {}, {}, lambda *_: None)


@pytest.mark.parametrize("error", [ValueError, MemoryError, AssertionError])
def test_selected_field_export_never_falls_back_to_exporting_every_field(error):
    from minflux_viewer.msr.session_export import subset_struct_fields

    class BrokenRecords(np.ndarray):
        __getitem__ = fail(error)

    records = np.zeros(2, dtype=[("x", "f8"), ("y", "f8")]).view(BrokenRecords)
    with pytest.raises(error, match="injected"):
        subset_struct_fields(records, {"x"})


@pytest.mark.parametrize("error", [ValueError, MemoryError, RuntimeError])
def test_saved_calibration_cannot_bypass_its_canonical_setter(error):
    from minflux_viewer.msr.viewer_state import STATE_FORMAT, apply_viewer_state

    dataset = SimpleNamespace(metadata={}, state={}, cali=SimpleNamespace(z_scaling_factor=1.0),
                              set_z_scaling_factor=fail(error))
    payload = {"format": STATE_FORMAT, "dataset": {"calibration": {"z_scaling_factor": 0.67}}}
    with pytest.raises(error, match="injected"):
        apply_viewer_state(dataset, payload)
    assert dataset.cali.z_scaling_factor == 1.0


@pytest.mark.parametrize("error", [MemoryError, AssertionError])
@pytest.mark.parametrize("escalation", [False, True])
def test_auto_fit_cannot_turn_resource_or_programming_error_into_another_winner(monkeypatch, error, escalation):
    from minflux_viewer.analysis import distribution_fit

    def dispatch(values, distribution, k, **kwargs):
        if not escalation or k == 2:
            raise error("injected")
        return SimpleNamespace(bic=100., n_components=1, distribution=distribution)

    monkeypatch.setattr(distribution_fit, "_dispatch_on_fitbasis", dispatch)
    with pytest.raises(error, match="injected"):
        distribution_fit.auto_fit(np.arange(10.), distributions=["gaussian"], max_components=1, hard_cap=2)


def test_auto_fit_retains_expected_bad_candidate_fallback(monkeypatch):
    from minflux_viewer.analysis import distribution_fit

    winner = SimpleNamespace(bic=10., n_components=1, distribution="gaussian")

    def dispatch(values, distribution, k, **kwargs):
        if k == 1:
            return winner
        raise ValueError("candidate cannot be fit")

    monkeypatch.setattr(distribution_fit, "_dispatch_on_fitbasis", dispatch)
    assert distribution_fit.auto_fit(np.arange(10.), distributions=["gaussian"], max_components=1, hard_cap=2) is winner
