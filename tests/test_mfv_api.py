"""
Extension-layer **track A** -- the published ``mfv`` API surface.

Covers the six namespaces track A owns (``data``, ``roi``, ``view``, ``ui``,
``run``, ``journal``), the facade that binds them, and the compatibility
aliases every previously published script name still resolves through.

``results`` and ``plot`` belong to track C and are only checked here for the
property track A depends on: that calling one raises a message naming its
owner rather than an AttributeError.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6.QtWidgets")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state(qapp):
    from minflux_viewer.core.app_state import AppState

    return AppState()


def _dataset(name="ds", *, n=40, seed=0, with_attrs=True):
    from minflux_viewer.core.dataset import build_localization_dataset

    rng = np.random.default_rng(seed)
    attrs = {}
    if with_attrs:
        attrs = {
            "efo": rng.uniform(1e4, 5e4, n),
            "cfr": rng.uniform(0.0, 1.0, n),
        }
    return build_localization_dataset(
        name=name,
        x_nm=rng.normal(0, 100, n),
        y_nm=rng.normal(0, 100, n),
        z_nm=rng.normal(0, 30, n),
        tid=np.repeat(np.arange(n // 4), 4)[:n],
        attrs=attrs,
    )


@pytest.fixture
def mfv(state):
    """The facade, with one dataset loaded and active."""
    state.add_dataset(_dataset())
    return state.mfv


# ---------------------------------------------------------------------------
# facade
# ---------------------------------------------------------------------------

def test_facade_exposes_every_published_namespace(mfv):
    from minflux_viewer.api import NAMESPACES

    for name in NAMESPACES:
        assert hasattr(mfv, name), name


def test_dataset_reference_accepts_index_name_and_object(mfv, state):
    ds = state.datasets[0]
    assert mfv.data.get() is ds            # None -> active
    assert mfv.data.get(0) is ds           # index
    assert mfv.data.get("ds") is ds        # name
    assert mfv.data.get(ds) is ds          # object
    assert mfv.data.get(np.int64(0)) is ds  # numpy index, as a script would give


def test_a_bad_dataset_reference_names_what_is_loaded(mfv):
    from minflux_viewer.scripting import ScriptError

    with pytest.raises(ScriptError) as excinfo:
        mfv.data.get("no such dataset")
    assert "'ds'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# mfv.data
# ---------------------------------------------------------------------------

def test_attr_returns_one_value_per_localization(mfv, state):
    ds = state.datasets[0]
    values = mfv.data.attr("efo")
    assert values.shape == (ds.prop.num_loc,)
    assert np.all(np.isfinite(values))


def test_attr_routes_through_the_iteration_aware_accessor(mfv, monkeypatch):
    """
    The rule that keeps a script agreeing with the Filter dialog: values come
    from ``attr_values_for_selection``, not from indexing ``ds.attr``.
    """
    import minflux_viewer.core.loader as loader

    seen = {}
    original = loader.attr_values_for_selection

    def spy(ds, attr, *, itr="auto"):
        seen["attr"] = attr
        seen["itr"] = itr
        return original(ds, attr, itr=itr)

    monkeypatch.setattr(loader, "attr_values_for_selection", spy)
    mfv.data.attr("cfr", itr="last")
    assert seen == {"attr": "cfr", "itr": "last"}


def test_attr_names_respects_preferences(mfv):
    names = mfv.data.attr_names()
    assert "idx" in names           # application-owned index is always first
    assert names[0] == "idx"


def test_unknown_attribute_lists_what_is_available(mfv):
    from minflux_viewer.scripting import ScriptError

    with pytest.raises(ScriptError) as excinfo:
        mfv.data.attr("not_an_attribute")
    assert "Available" in str(excinfo.value)


def test_loc_units_and_filtering(mfv, state):
    ds = state.datasets[0]
    nm = mfv.data.loc()
    m = mfv.data.loc(unit="m")
    assert nm.shape == (ds.prop.num_loc, 3)
    assert np.allclose(nm / 1e9, m)


def test_loc_rejects_an_unknown_unit(mfv):
    from minflux_viewer.scripting import ScriptError

    with pytest.raises(ScriptError):
        mfv.data.loc(unit="micron")


def test_add_attr_lands_in_derived_and_is_offered(mfv, state):
    ds = state.datasets[0]
    values = np.arange(ds.prop.num_loc, dtype=float)
    mfv.data.add_attr("score", values)
    assert np.allclose(np.asarray(ds.derived["score"]), values)
    assert "score" in mfv.data.attr_names()


def test_add_attr_rejects_a_wrong_length_and_says_so(mfv):
    from minflux_viewer.scripting import ScriptError

    with pytest.raises(ScriptError) as excinfo:
        mfv.data.add_attr("bad", np.zeros(3))
    assert "one value per localization" in str(excinfo.value)


def test_create_adds_a_dataset_with_the_z_factor_pinned(mfv, state):
    """
    Computed coordinates are final. Pinning the factor and recording it is what
    stops the post-load chain estimating one and scaling z a second time.
    """
    before = len(state.datasets)
    pts = np.column_stack([np.arange(10.0), np.arange(10.0), np.zeros(10)])
    ds = mfv.data.create(pts, name="computed")

    assert len(state.datasets) == before + 1
    assert ds.cali.z_scaling_factor == 1.0
    assert ds.derived["z_scaling_factor"] == 1.0
    assert ds.metadata["created_by_script"] is True


def test_create_rejects_a_bad_shape(mfv):
    from minflux_viewer.scripting import ScriptError

    with pytest.raises(ScriptError):
        mfv.data.create(np.zeros((10, 5)))


def test_set_filter_applies_and_persists_a_reevaluable_spec(mfv, state):
    ds = state.datasets[0]
    efo = mfv.data.attr("efo", filtered=False)
    cut = float(np.median(efo))

    mask = mfv.data.set_filter([
        {"attribute": "efo", "mode": "per loc", "lo": cut, "hi": float(efo.max())},
    ])

    assert mask.sum() < ds.prop.num_loc
    assert np.array_equal(mask, mfv.data.filter_mask())
    # persisted in the re-evaluable form the Filter dialog writes
    spec = mfv.data.filter_specs()[0]
    assert spec["attribute"] == "efo" and spec["mode"] == "per loc"
    assert "itr" in spec and "lo_inc" in spec
    # and it actually filters what a reader returns
    assert mfv.data.attr("efo").size == int(mask.sum())


def test_set_filter_can_append_instead_of_replacing(mfv):
    efo = mfv.data.attr("efo", filtered=False)
    mfv.data.set_filter([{"attribute": "efo", "mode": "per loc",
                          "lo": float(efo.min()), "hi": float(efo.max())}])
    mfv.data.set_filter([{"attribute": "cfr", "mode": "per loc",
                          "lo": 0.0, "hi": 0.5}], replace=False)
    assert [s["attribute"] for s in mfv.data.filter_specs()] == ["efo", "cfr"]


def test_set_filter_rejects_an_unknown_mode(mfv):
    from minflux_viewer.scripting import ScriptError

    with pytest.raises(ScriptError) as excinfo:
        mfv.data.set_filter([{"attribute": "efo", "mode": "nonsense",
                              "lo": 0, "hi": 1}])
    assert "mode" in str(excinfo.value)


def test_clear_filter_restores_every_localization(mfv, state):
    ds = state.datasets[0]
    efo = mfv.data.attr("efo", filtered=False)
    mfv.data.set_filter([{"attribute": "efo", "mode": "per loc",
                          "lo": float(np.median(efo)), "hi": float(efo.max())}])
    mfv.data.clear_filter()
    assert mfv.data.filter_mask().all()
    assert mfv.data.filter_specs() == []
    assert mfv.data.attr("efo").size == ds.prop.num_loc


# ---------------------------------------------------------------------------
# mfv.roi
# ---------------------------------------------------------------------------

def _rect_over_everything(mfv):
    xyz = mfv.data.loc()
    x0, y0 = xyz[:, 0].min() - 1, xyz[:, 1].min() - 1
    w = np.ptp(xyz[:, 0]) + 2
    h = np.ptp(xyz[:, 1]) + 2
    return mfv.roi.add("rectangle", {"bounds": [float(x0), float(y0), float(w), float(h)]})


def test_a_scripted_roi_is_scoped_to_its_dataset_and_view(mfv):
    """
    An unscoped record is drawn in every view of every dataset, so the scope
    must be stamped at creation -- a line or point never runs a selection pass
    and would otherwise stay unscoped forever.
    """
    record = mfv.roi.add("rectangle", {"bounds": [0.0, 0.0, 10.0, 10.0]})
    assert record.context["dataset_idx"] == 0
    assert record.context["source_view"] == "render"


def test_roi_add_rejects_an_unknown_type(mfv):
    from minflux_viewer.scripting import ScriptError

    with pytest.raises(ScriptError):
        mfv.roi.add("blob", {"bounds": [0, 0, 1, 1]})


def test_roi_list_and_remove(mfv):
    record = mfv.roi.add("rectangle", {"bounds": [0.0, 0.0, 5.0, 5.0]})
    assert record in mfv.roi.list()
    assert mfv.roi.remove(record) is True
    assert record not in mfv.roi.list()
    assert mfv.roi.remove(record) is False


def test_mask_and_points_in_agree_and_align_with_attributes(mfv):
    """
    Row alignment between coordinates, masks and attributes is what makes
    ``attr_in`` meaningful.
    """
    record = _rect_over_everything(mfv)
    mask = mfv.roi.mask(record)
    points = mfv.roi.points_in(record)
    values = mfv.roi.attr_in(record, "efo")

    assert mask.all()                       # the box covers the whole dataset
    assert points.shape[0] == int(mask.sum())
    assert values.shape[0] == int(mask.sum())


def test_a_half_covering_roi_selects_a_subset(mfv):
    xyz = mfv.data.loc()
    mid = float(np.median(xyz[:, 0]))
    record = mfv.roi.add("rectangle", {
        "bounds": [mid, float(xyz[:, 1].min() - 1),
                   float(xyz[:, 0].max() - mid + 1), float(np.ptp(xyz[:, 1]) + 2)],
    })
    mask = mfv.roi.mask(record)
    assert 0 < mask.sum() < mask.size


def test_an_open_line_roi_encloses_nothing_rather_than_erroring(mfv):
    record = mfv.roi.add("line", {"points": [[0.0, 0.0], [10.0, 10.0]]})
    assert not mfv.roi.mask(record).any()


def test_crop_refuses_a_non_region_roi_and_names_the_valid_types(mfv):
    from minflux_viewer.scripting import ScriptError

    record = mfv.roi.add("line", {"points": [[0.0, 0.0], [1.0, 1.0]]})
    with pytest.raises(ScriptError) as excinfo:
        mfv.roi.crop(record)
    assert "rectangle" in str(excinfo.value)


def test_crop_produces_a_smaller_dataset(mfv, state):
    xyz = mfv.data.loc()
    mid = float(np.median(xyz[:, 0]))
    record = mfv.roi.add("rectangle", {
        "bounds": [mid, float(xyz[:, 1].min() - 1),
                   float(xyz[:, 0].max() - mid + 1), float(np.ptp(xyz[:, 1]) + 2)],
    })
    before = len(state.datasets)
    cropped = mfv.roi.crop(record, exact_shape=True)
    assert len(state.datasets) == before + 1
    assert cropped.prop.num_loc < state.datasets[0].prop.num_loc


def test_roi_bounds_are_x_y_w_h(mfv):
    record = mfv.roi.add("rectangle", {"bounds": [1.0, 2.0, 30.0, 40.0]})
    assert mfv.roi.bounds(record) == (1.0, 2.0, 30.0, 40.0)


# ---------------------------------------------------------------------------
# mfv.journal
# ---------------------------------------------------------------------------

def test_journal_record_reaches_the_journal_and_the_log(mfv, state):
    mfv.journal.record("analysis", "Measured something", dataset=0, radius_nm=25)
    entry = list(state.journal)[-1]
    assert entry.category == "analysis"
    assert entry.details["radius_nm"] == 25
    # also in the Log, tagged to the dataset, so it is visible during the run
    last = state.log_history[-1]
    assert "Measured something" in last["message"]
    assert "radius_nm=25" in last["message"]


def test_journal_rejects_an_unknown_category(mfv):
    from minflux_viewer.scripting import ScriptError

    with pytest.raises(ScriptError):
        mfv.journal.record("nonsense", "x")


# ---------------------------------------------------------------------------
# mfv.ui
# ---------------------------------------------------------------------------

def test_ui_log_tags_the_dataset(mfv, state):
    mfv.ui.log("hello", "WARN", dataset=0)
    last = state.log_history[-1]
    assert last["message"] == "hello" and last["level"] == "WARN"
    assert last["dataset_idx"] == 0


def test_ui_status_reaches_the_status_line(mfv, state):
    seen = []
    state.status_message.connect(seen.append)
    mfv.ui.status("working", 0.5)
    assert seen and "working" in seen[-1]


# ---------------------------------------------------------------------------
# mfv.run
# ---------------------------------------------------------------------------

def test_background_work_runs_off_thread_and_delivers_on_the_gui_thread(mfv, qapp):

    out = {}

    def work(task):
        task.progress(1, 2, "half")
        return 6 * 7

    mfv.run.background(work, on_done=lambda v: out.setdefault("value", v),
                       name="unit test task")

    from minflux_viewer.ui.background_tasks import shared_thread_pool

    shared_thread_pool("mfv-script").waitForDone(5000)
    for _ in range(50):
        qapp.processEvents()
        if "value" in out:
            break
    assert out.get("value") == 42


def test_background_failure_is_reported_not_swallowed(mfv, qapp):
    from minflux_viewer.ui.background_tasks import shared_thread_pool

    errors = []

    def work(task):
        raise ValueError("deliberate")

    mfv.run.background(work, on_error=errors.append, name="failing task")
    shared_thread_pool("mfv-script").waitForDone(5000)
    for _ in range(50):
        qapp.processEvents()
        if errors:
            break
    assert errors and "deliberate" in str(errors[0])


def test_background_rejects_a_non_callable(mfv):
    from minflux_viewer.scripting import ScriptError

    with pytest.raises(ScriptError):
        mfv.run.background(object())


# ---------------------------------------------------------------------------
# compatibility -- every previously published name still works
# ---------------------------------------------------------------------------

def test_legacy_top_level_names_still_resolve(mfv, state):
    ds = state.datasets[0]
    assert mfv.get_active_dataset() is ds
    assert mfv.get_datasets() == [ds]
    assert mfv.get_dataset(0) is ds
    assert mfv.get_attr("efo").shape == (ds.prop.num_loc,)
    assert mfv.get_loc().shape == (ds.prop.num_loc, 3)
    mfv.log("still works")
    assert state.log_history[-1]["message"] == "still works"


def test_legacy_get_attr_other_sources_still_read_components(mfv, state):
    ds = state.datasets[0]
    ds.derived["thing"] = np.ones(ds.prop.num_loc)
    assert mfv.get_attr("thing", source="derived").size == ds.prop.num_loc


def test_runtime_module_publishes_namespaces_and_legacy_names(mfv):
    from minflux_viewer.api import NAMESPACES
    from minflux_viewer.scripting import install_runtime_module

    module = install_runtime_module(mfv)
    for name in NAMESPACES:
        assert hasattr(module, name), name
    for name in ("get_active_dataset", "get_loc", "log", "viewer", "ScriptError"):
        assert hasattr(module, name), name
    assert module.__api_version__ == "1.0"


def test_script_error_is_the_api_error(mfv):
    from minflux_viewer.api._base import ApiError
    from minflux_viewer.scripting import ScriptError

    assert ScriptError is ApiError


# ---------------------------------------------------------------------------
# the seam with track C
# ---------------------------------------------------------------------------

def test_track_a_can_drive_the_results_and_plot_namespaces(mfv):
    """
    The seam between the two tracks, exercised from Track A's side: a script
    reads through ``mfv.data`` and writes through ``mfv.results`` / ``mfv.plot``
    without either namespace knowing about the other.
    """
    efo = mfv.data.attr("efo")

    table = mfv.results.table("Track A seam")
    table.add_row(n=int(efo.size), median_efo=float(np.median(efo)))
    assert len(table) == 1
    assert table.to_dict()["n"] == [int(efo.size)]

    handle = mfv.plot.hist(efo, bins=10, title="efo")
    # chainable, and still the same handle
    assert handle.labels(x="efo (Hz)", y="count").legend(True) is handle


def test_deferred_members_explain_themselves(mfv):
    """
    ``volume`` and ``lut`` are published names deliberately not implemented in
    API 1.0. A caller must get the reason, not an AttributeError.
    """
    for call in (mfv.view.volume, mfv.view.lut):
        with pytest.raises(NotImplementedError) as excinfo:
            call()
        assert "1.0" in str(excinfo.value)
