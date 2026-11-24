"""One acquisition, three containers: ``.npy``, ``.mat`` and ``.json`` must load
into the same dataset.

The three files hold identical values -- Imspector writes the same m2410 record
table into each -- so after loading only the container may differ. Each aspect
is its own test. They used to differ (JSON reported "json" as its version and
lost the effective CFR/EFC iteration, so CFR read as zero wherever it was
measured before the last iteration; .mat kept MATLAB's uint8 flags and dropped
the second dcr channel) until every flat reader was built by
``load_from_mfx_columns``.

The synthetic acquisition is built to exercise what differed in practice: CFR
measured only at an intermediate iteration (so the effective-iteration
metadata matters), a two-column ``dcr``, invalid rows, and Imspector's own
dtypes (bool flags, ``u1``/``u4`` integers, ``f2`` cfr/dcr).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.lib.recfunctions as rfn
import pytest
import scipy.io as sio

from minflux_viewer.core.app_state import default_prefs
from minflux_viewer.core.loader_format_readers import load_dataset, load_json, load_npy
from minflux_viewer.core.loader_raw_access import attr_values_1d
from minflux_viewer.core.save import _write_json

FORMATS = ("npy", "mat", "json")
SAMPLE_DIR = Path(r"D:\Workspace\Microscopes\MINFLUX\sample data\raw_data_formats")
SAMPLE_STEM = "240624-174045_minflux"

#: Imspector's m2410 record layout (ZARR files version m2410).
M2410_DTYPE = np.dtype([
    ("vld", "?"), ("fnl", "?"), ("bot", "?"), ("eot", "?"), ("sta", "u1"),
    ("tim", "<f8"), ("tid", "<u4"), ("gri", "<u4"), ("thi", "u1"), ("sqi", "u1"),
    ("itr", "<i4"), ("loc", "<f8", (3,)), ("lnc", "<f8", (3,)), ("eco", "<u4"),
    ("ecc", "<u4"), ("efo", "<f4"), ("efc", "<f4"), ("fbg", "<f4"), ("cfr", "<f2"),
    ("dcr", "<f2", (2,)),
])


def synthetic_m2410(n_loc: int = 40, n_itr: int = 5, cfr_itrs=(2, 3), *,
                    last_itr_valid: bool = True, seed: int = 0) -> np.ndarray:
    """An m2410 table: every localization runs all iterations; a few fail."""
    rng = np.random.default_rng(seed)
    n = n_loc * n_itr
    mfx = np.zeros(n, dtype=M2410_DTYPE)
    itr = np.tile(np.arange(n_itr), n_loc)
    loc_index = np.repeat(np.arange(n_loc), n_itr)
    mfx["itr"] = itr
    mfx["tid"] = (loc_index // 4).astype(np.uint32)
    mfx["gri"] = mfx["tid"]
    mfx["tim"] = np.arange(n) * 1e-3
    vld = rng.random(n) > 0.1
    if not last_itr_valid:
        vld[itr == n_itr - 1] = False
    mfx["vld"] = vld
    final = itr == n_itr - 1
    mfx["fnl"] = vld & final
    mfx["sta"] = np.where(vld & final, 1, np.where(vld, 2, 8)).astype(np.uint8)
    xy = rng.uniform(0.0, 2e-6, (n_loc, 2))
    mfx["loc"][:, :2] = np.repeat(xy, n_itr, axis=0) + rng.normal(0, 5e-9, (n, 2))
    mfx["lnc"] = mfx["loc"]
    mfx["eco"] = rng.integers(50, 300, n).astype(np.uint32)
    mfx["efo"] = rng.uniform(2e4, 9e4, n).astype(np.float32)
    centre = np.isin(itr, cfr_itrs)
    mfx["efc"][centre] = rng.uniform(1e3, 1e4, int(centre.sum())).astype(np.float32)
    mfx["ecc"][centre] = rng.integers(1, 30, int(centre.sum())).astype(np.uint32)
    mfx["cfr"][centre] = rng.uniform(0.05, 0.6, int(centre.sum())).astype(np.float16)
    ratio = rng.uniform(0.2, 0.8, n).astype(np.float16)
    mfx["dcr"][:, 0] = ratio
    mfx["dcr"][:, 1] = (1.0 - ratio.astype(np.float32)).astype(np.float16)
    mfx["fbg"] = rng.uniform(500, 3000, n).astype(np.float32)
    return mfx


def write_three(mfx: np.ndarray, folder: Path, stem: str = "acq") -> dict[str, Path]:
    """The table as Imspector-style ``.npy``, top-level-variable ``.mat`` and record ``.json``."""
    paths = {fmt: folder / f"{stem}.{fmt}" for fmt in FORMATS}
    np.save(paths["npy"], mfx, allow_pickle=False)
    sio.savemat(str(paths["mat"]), {name: mfx[name] for name in mfx.dtype.names})
    _write_json(paths["json"], mfx)
    return paths


def load_three(paths: dict[str, Path]) -> dict:
    prefs = default_prefs()
    return {
        "npy": load_npy(paths["npy"], prefs=prefs, apply_sidecar=False),
        "mat": load_dataset(paths["mat"], prefs=prefs, apply_sidecar=False),
        "json": load_json(paths["json"], prefs=prefs, apply_sidecar=False),
    }


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    return load_three(write_three(synthetic_m2410(), tmp_path_factory.mktemp("equivalence")))


def _floats(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


# ----------------------------------------------------------- values and counts
def test_raw_store_values_agree(loaded):
    ref = loaded["npy"].components.mfx_raw
    for fmt in ("mat", "json"):
        other = loaded[fmt].components.mfx_raw
        assert set(other.keys()) == set(ref.keys()), fmt
        for key in ref.keys():
            np.testing.assert_array_equal(_floats(other[key]), _floats(ref[key]),
                                          err_msg=f"{fmt}: {key}")


def test_materialized_values_agree(loaded):
    ref = loaded["npy"].attr
    for fmt in ("mat", "json"):
        other = loaded[fmt].attr
        for key in sorted(set(ref.keys()) & set(other.keys())):
            np.testing.assert_array_equal(_floats(other[key]), _floats(ref[key]),
                                          err_msg=f"{fmt}: {key}")


def test_counts_agree(loaded):
    for key in ("raw_num_loc", "valid_num_loc", "raw_num_itr"):
        values = {fmt: loaded[fmt].metadata.get(key) for fmt in FORMATS}
        assert len(set(values.values())) == 1, (key, values)


@pytest.mark.parametrize("status", ["no status column", "final rows at the 4th iteration"])
@pytest.mark.parametrize("fmt", FORMATS)
def test_raw_iteration_count_includes_an_all_invalid_last_iteration(tmp_path, fmt, status):
    """Counted from every raw row, never from the valid ones: the canonical
    builder used to report 4 here while the .mat/.json readers said 5.

    With every 5th-iteration row invalid there are localizations only if they
    end earlier: without a status column (the older rule takes the highest
    valid iteration), or with ``sta == 1`` marking them at the 4th iteration.
    """
    mfx = synthetic_m2410(last_itr_valid=False)
    if status == "no status column":
        mfx = rfn.drop_fields(mfx, "sta", usemask=False)
    else:
        mfx["sta"] = np.where(mfx["vld"] & (mfx["itr"] == 3), 1,
                              np.where(mfx["vld"], 2, 8)).astype(np.uint8)
    ds = load_three(write_three(mfx, tmp_path))[fmt]
    assert ds.metadata["raw_num_itr"] == 5
    assert ds.prop.num_loc == int((mfx["vld"] & (mfx["itr"] == 3)).sum()) > 0


# ------------------------------------------- what used to differ by container
@pytest.mark.parametrize("fmt", FORMATS)
def test_structural_version_and_effective_iterations(loaded, fmt):
    md = loaded[fmt].metadata
    assert md.get("source_version") == "m2410"
    assert str(md.get("source_version_detail", "")).startswith("m2410")
    assert md.get("cfr_iteration") == 3          # 1-based: measured last at index 2..3
    assert md.get("efc_iteration") == 3


@pytest.mark.parametrize("fmt", FORMATS)
def test_cfr_is_read_at_its_effective_iteration(loaded, fmt):
    """Every histogram, filter row and Auto rule reads ``attr_values_1d``."""
    cfr = _floats(attr_values_1d(loaded[fmt], "cfr"))
    reference = _floats(attr_values_1d(loaded["npy"], "cfr"))
    assert np.count_nonzero(cfr) > 0
    np.testing.assert_array_equal(cfr, reference)


@pytest.mark.parametrize("fmt", FORMATS)
def test_both_dcr_channels_are_attributes(loaded, fmt):
    assert {"dcr", "dcr_0", "dcr_1"} <= set(loaded[fmt].attr.keys())


@pytest.mark.parametrize("fmt", FORMATS)
def test_flags_are_boolean(loaded, fmt):
    for key in ("vld", "fnl", "bot", "eot"):
        assert np.asarray(loaded[fmt].attr[key]).dtype == np.bool_, key
        assert np.asarray(loaded[fmt].components.mfx_raw[key]).dtype == np.bool_, key


@pytest.mark.parametrize("fmt", ["mat", "json"])
def test_attribute_order_agrees(loaded, fmt):
    assert list(loaded[fmt].prop.attr_names) == list(loaded["npy"].prop.attr_names)


@pytest.mark.parametrize("fmt", FORMATS)
def test_container_is_recorded_beside_the_version(loaded, fmt):
    assert loaded[fmt].metadata.get("source_format") == fmt


def test_a_one_value_spatial_field_is_not_split_into_invented_components():
    """Some exports store ``ext`` as one value per row (the pattern extent).
    Splitting it as if it were an x/y/z vector invented y and z columns of
    zeros; a one-value field keeps its own name."""
    from minflux_viewer.core.loader_format_readers import load_from_mfx_columns

    mfx = synthetic_m2410()
    columns = {name: np.asarray(mfx[name]) for name in mfx.dtype.names}
    columns["ext"] = np.full(len(mfx), 1.008e-07)
    ds = load_from_mfx_columns(columns, "ext")
    assert "ext" in ds.attr.keys()
    assert not {"ext_x", "ext_y", "ext_z"} & set(ds.attr.keys())
    np.testing.assert_array_equal(ds.attr["ext"], 1.008e-07)


# ----------------------------------------------------- the real sample triple
@pytest.fixture(scope="module")
def sample_triple():
    paths = {fmt: SAMPLE_DIR / f"{SAMPLE_STEM}.{fmt}" for fmt in FORMATS}
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("Abberior raw-format sample triple not available")
    return load_three(paths)


def test_sample_triple_values_agree(sample_triple):
    ref = sample_triple["npy"]
    for fmt in ("mat", "json"):
        for key in sorted(set(ref.attr.keys()) & set(sample_triple[fmt].attr.keys())):
            np.testing.assert_array_equal(_floats(sample_triple[fmt].attr[key]),
                                          _floats(ref.attr[key]), err_msg=f"{fmt}: {key}")


@pytest.mark.parametrize("fmt", FORMATS)
def test_sample_triple_reports_m2410_with_cfr_iteration(sample_triple, fmt):
    md = sample_triple[fmt].metadata
    assert md.get("source_version") == "m2410"
    assert md.get("cfr_iteration") == 5 and md.get("efc_iteration") == 5
