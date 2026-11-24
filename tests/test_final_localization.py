"""The final localization is the file's own ``sta == 1`` ("valid final iteration").

A measurement can run several threads, each with its own sequence; when they end
at different iterations, keeping only the rows at the single highest iteration
dropped every final localization of the shorter thread (41 % of the valid final
localizations in one local two-thread acquisition). The file's per-row status
keeps them all; on single-thread data it selects exactly the rows the older
rule did.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.core.final_localization import final_rows
from tests.test_format_equivalence import (
    FORMATS,
    M2410_DTYPE,
    load_three,
    synthetic_m2410,
    write_three,
)

SAMPLE = Path(r"D:\Workspace\Microscopes\MINFLUX\sample data"
              r"\250702-172925_03_L75_561lp06_640lp10_pho010010_BGdis_seq.mat")


def two_thread_m2410(seed: int = 1) -> np.ndarray:
    """Thread 0 runs 4 iterations, thread 1 runs 3; their traces are distinct."""
    rng = np.random.default_rng(seed)
    parts = []
    tid_offset = 0
    for thread, n_itr, n_loc in ((0, 4, 30), (1, 3, 20)):
        n = n_itr * n_loc
        part = np.zeros(n, dtype=M2410_DTYPE)
        itr = np.tile(np.arange(n_itr), n_loc)
        part["itr"] = itr
        part["thi"] = thread
        part["tid"] = tid_offset + np.repeat(np.arange(n_loc) // 3, n_itr)
        vld = rng.random(n) > 0.15
        final = itr == n_itr - 1
        part["vld"] = vld
        part["fnl"] = final
        part["sta"] = np.where(vld & final, 1, np.where(vld, 2, 8))
        part["loc"][:, :2] = np.repeat(rng.uniform(0, 2e-6, (n_loc, 2)), n_itr, axis=0)
        part["lnc"] = part["loc"]
        part["efo"] = rng.uniform(2e4, 9e4, n)
        part["dcr"][:, 0] = 0.5
        part["dcr"][:, 1] = 0.5
        parts.append(part)
        tid_offset = int(part["tid"].max()) + 1
    mfx = np.concatenate(parts)
    mfx["tim"] = np.arange(len(mfx)) * 1e-3
    return mfx


# ------------------------------------------------------------------ the rule
def test_with_a_status_every_threads_final_rows_are_kept():
    mfx = two_thread_m2410()
    rows = final_rows(mfx["itr"], mfx["vld"], mfx["sta"], mfx["fnl"])
    np.testing.assert_array_equal(rows, mfx["sta"] == 1)
    assert set(mfx["itr"][rows & (mfx["thi"] == 0)]) == {3}
    assert set(mfx["itr"][rows & (mfx["thi"] == 1)]) == {2}


def test_on_single_thread_data_the_status_and_the_old_rule_agree():
    mfx = synthetic_m2410()
    with_status = final_rows(mfx["itr"], mfx["vld"], mfx["sta"], mfx["fnl"])
    without = final_rows(mfx["itr"], mfx["vld"])
    np.testing.assert_array_equal(with_status, without)


def test_including_invalid_rows_takes_every_final_event():
    mfx = two_thread_m2410()
    rows = final_rows(mfx["itr"], mfx["vld"], mfx["sta"], mfx["fnl"], vld_only=False)
    np.testing.assert_array_equal(rows, mfx["fnl"] | (mfx["sta"] == 1))


def test_without_a_status_the_highest_valid_iteration_stands():
    itr = np.array([0, 1, 2, 0, 1, 2])
    vld = np.array([True, True, False, True, True, True])
    np.testing.assert_array_equal(final_rows(itr, vld), [False, False, False, False, False, True])
    assert not final_rows(itr, np.zeros(6, bool)).any()


def test_a_status_with_no_final_row_means_no_localization():
    """A documented status column is authoritative even when no row is final:
    ``sta`` 2 is "valid, not final", so the older rule must not turn the
    highest such row into a localization (final review, finding 6)."""
    itr, vld = np.array([0, 1]), np.array([True, True])
    assert not final_rows(itr, vld, np.array([2, 2]), np.array([False, False])).any()
    # Including invalid rows: still only what the file flags as final events.
    assert not final_rows(itr, vld, np.array([2, 8]), np.array([False, False]),
                          vld_only=False).any()


def test_a_column_of_zeros_is_not_a_status_record():
    """No documented code: not a status, so the older rule applies."""
    itr, vld = np.array([0, 1]), np.array([True, True])
    np.testing.assert_array_equal(final_rows(itr, vld, np.zeros(2, np.uint8)), [False, True])


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_reader_given_a_status_without_final_rows_loads_none(tmp_path, fmt):
    """``.npy``/``.mat`` give an empty dataset; the JSON reader has always
    refused a file without localizations, and still does."""
    from minflux_viewer.core.loader_format_readers import load_dataset, load_json, load_npy

    mfx = synthetic_m2410(last_itr_valid=False)           # sta 2 or 8, never 1
    assert not (mfx["sta"] == 1).any()
    path = write_three(mfx, tmp_path)[fmt]
    if fmt == "json":
        with pytest.raises(ValueError, match="no valid finite localizations"):
            load_json(path, apply_sidecar=False)
        return
    ds = {"npy": load_npy, "mat": load_dataset}[fmt](path, apply_sidecar=False)
    assert ds.prop.num_loc == 0 and ds.metadata["raw_num_itr"] == 5


# --------------------------------------------------------- through the readers
@pytest.fixture(scope="module")
def two_thread(tmp_path_factory):
    return load_three(write_three(two_thread_m2410(), tmp_path_factory.mktemp("threads")))


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_reader_keeps_both_threads(two_thread, fmt):
    from minflux_viewer.core.loader_raw_access import attr_matches_last_valid, mfx_get

    mfx = two_thread_m2410()
    ds = two_thread[fmt]
    assert ds.prop.num_loc == int(np.count_nonzero(mfx["sta"] == 1))
    assert set(np.asarray(ds.attr["thi"]).astype(int)) == {0, 1}
    assert ds.metadata["raw_num_itr"] == 4
    # Raw browsing selects the same rows the dataset was built from.
    assert attr_matches_last_valid(ds)
    np.testing.assert_array_equal(np.asarray(mfx_get(ds, "tid", itr="last")),
                                  np.asarray(ds.attr["tid"]))


def test_the_three_readers_agree_on_a_two_thread_acquisition(two_thread):
    ref = two_thread["npy"].attr
    for fmt in ("mat", "json"):
        for key in ("loc_x", "tid", "itr", "thi", "sta"):
            np.testing.assert_array_equal(np.asarray(two_thread[fmt].attr[key], dtype=float),
                                          np.asarray(ref[key], dtype=float), err_msg=key)


def test_a_mapped_table_with_a_status_column_follows_the_same_rule():
    from minflux_viewer.core.spreadsheet_loader import minflux_row_mask

    mfx = two_thread_m2410()
    rows = minflux_row_mask(mfx["itr"].astype(float), mfx["vld"].astype(float),
                            mfx["sta"].astype(float))
    np.testing.assert_array_equal(rows, mfx["sta"] == 1)
    old_rule = minflux_row_mask(mfx["itr"].astype(float), mfx["vld"].astype(float))
    assert old_rule.sum() < rows.sum()            # the older rule loses thread 1


def test_the_local_two_thread_acquisition_keeps_its_second_thread():
    if not SAMPLE.is_file():
        pytest.skip("local two-thread sample not available")
    from minflux_viewer.core.app_state import default_prefs
    from minflux_viewer.core.loader_format_readers import load_dataset

    ds = load_dataset(SAMPLE, prefs=default_prefs(), apply_sidecar=False)
    assert ds.prop.num_loc == 124_901                     # was 73,182 (thread 0 only)
    thi = np.asarray(ds.attr["thi"]).astype(int)
    assert int(np.count_nonzero(thi == 1)) == 51_719
