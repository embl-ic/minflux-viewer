"""Sample-data evidence for raw modern orbit counts, not instrument-wide proof.

The signal at a centred quadratic minimum scales as power times pattern size
squared. Paired iterations fit that scaling after subtracting the measured
background from EFO; treating EFO as already background-subtracted does not.
This constrains ECO/EFO for these m2410 acquisitions, not the unresolved
ECC/EFC convention with background sensing enabled.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.core.loader import iteration_attr_values_1d
from minflux_viewer.core.mfx_sequence import background_sensing, sequence_for_dataset


_SAMPLE = Path(r"D:\Workspace\Microscopes\MINFLUX\sample data")
_CASES = (
    "3_Bonly_MRED_75pM_MINFLUX_3D.msr",
    "10_Nup62_215TCO_LD655_50nM_2mM_THP_anti-GFP-PAINT_Ab580_125pM.msr",
    "2_3C_measurement.msr",
)


@pytest.fixture(scope="module", params=_CASES, ids=("Bonly", "Nup62", "2_3C"))
def sample_case(request):
    path = _SAMPLE / request.param
    if not path.is_file():
        pytest.skip(f"sample MSR not available: {path}")
    from minflux_viewer.core.app_state import DEFAULT_PREFS
    from minflux_viewer.msr.quick_open import load_viewer_msr

    datasets = load_viewer_msr(path, prefs=deepcopy(DEFAULT_PREFS))
    if path.name.startswith("10_"):
        # The second run is the sensing-on ten-iteration acquisition; identify
        # it from the sequence, not the reader's presentation order.
        datasets = [ds for ds in datasets
                    if background_sensing(sequence_for_dataset(ds)) is True]
        assert len(datasets) == 1
    else:
        assert len(datasets) == (3 if path.name.startswith("2_") else 1)
    yield path.name, datasets


def _at(dataset, field, iteration):
    values = iteration_attr_values_1d(dataset, field, iteration,
                                    itr="last", vld_only=True)
    assert values is not None, (dataset.name, field, iteration)
    # Copy so the later CRLB-input check also detects accidental in-place edits.
    return np.asarray(values).copy()


def _paired_scaling(dataset, first, final, background_field):
    fields = ("efo", background_field, "vld")
    earlier = {field: _at(dataset, field, first) for field in fields}
    later = {field: _at(dataset, field, final) for field in fields}
    shape = earlier["efo"].shape
    assert all(values.shape == shape for values in (*earlier.values(), *later.values()))
    valid = (earlier["vld"] == 1) & (later["vld"] == 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_ratio = ((later["efo"] - later[background_field])
                     / (earlier["efo"] - earlier[background_field]))
        net_ratio = later["efo"] / earlier["efo"]
    paired = valid & np.isfinite(raw_ratio) & np.isfinite(net_ratio)
    assert int(paired.sum()) > 10_000
    iterations = sequence_for_dataset(dataset)["Itr"]
    start, end = iterations[first], iterations[final]
    expected = (end["pwrFactor"] / start["pwrFactor"]
                * (end["patGeoFactor"] / start["patGeoFactor"]) ** 2)
    return expected, float(np.median(raw_ratio[paired])), float(np.median(net_ratio[paired]))


def test_modern_orbit_rates_fit_raw_count_scaling(sample_case):
    name, datasets = sample_case
    is_control = name.startswith("2_")
    pairs = ((3, 4),) if is_control else ((6, 8), (7, 9))
    raw_limits = ((0.37, 0.43) if is_control else
                  (0.40, 0.50) if name.startswith("10_") else (0.38, 0.44))
    for dataset in datasets:
        assert dataset.metadata["source_version"] == "m2410"
        sensing = background_sensing(sequence_for_dataset(dataset))
        assert sensing is (not is_control)
        background_field = "efc" if is_control else "fbg"
        for first, final in pairs:
            expected, raw, net = _paired_scaling(dataset, first, final, background_field)
            assert expected == pytest.approx(1.5 * (0.11 / 0.21) ** 2)
            assert raw_limits[0] < raw < raw_limits[1], (dataset.name, first, final, raw)
            assert 0.65 < net < 1.05, (dataset.name, first, final, net)
            assert abs(raw - expected) < abs(net - expected) / 2


def test_crlb_does_not_restore_background_into_modern_orbit_counts(sample_case):
    from minflux_viewer.analysis.localization_precision import crlb_inputs_for_dataset

    name, datasets = sample_case
    axes = (("xy", 4),) if name.startswith("2_") else (("xy", 8), ("z", 9))
    for dataset in datasets:
        recorded = {(axis, field): _at(dataset, field, iteration)
                    for axis, iteration in axes for field in ("eco", "efo")}
        inputs = crlb_inputs_for_dataset(dataset)
        assert inputs is not None
        for axis, iteration in axes:
            for field in ("eco", "efo"):
                np.testing.assert_array_equal(inputs[axis][field], recorded[axis, field])
                np.testing.assert_array_equal(_at(dataset, field, iteration), recorded[axis, field])
