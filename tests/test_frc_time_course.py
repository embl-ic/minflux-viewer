"""FRC over acquisition time (``precision_calculation.frc_time_course``)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from minflux_viewer.analysis import precision_calculation as pc


def _structure(seed: int = 0, n_emitters: int = 600, locs_per_emitter: int = 8,
               sigma_nm: float = 4.0, duration_s: float = 600.0):
    """Emitters in a 2 µm field, each localized a few times at its own moment."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.0, 2000.0, size=(n_emitters, 2))
    onset = rng.uniform(0.0, duration_s, n_emitters)
    idx = np.repeat(np.arange(n_emitters), locs_per_emitter)
    xy = centres[idx] + rng.normal(0.0, sigma_nm, size=(idx.size, 2))
    t = onset[idx] + rng.uniform(0.0, 0.5, idx.size)
    return xy[:, 0], xy[:, 1], t, idx


def test_window_ends_are_cumulative_and_end_at_the_last_localization():
    t = np.array([5.0, 7.0, 15.0, 25.0])
    np.testing.assert_allclose(pc.frc_window_ends(t, n_steps=4), [10.0, 15.0, 20.0, 25.0])
    np.testing.assert_allclose(pc.frc_window_ends(t, step_s=8.0), [13.0, 21.0, 25.0])
    np.testing.assert_allclose(pc.frc_window_ends(np.array([3.0, 3.0])), [3.0])
    assert pc.frc_window_ends(np.array([np.nan])).size == 0


def test_last_window_reproduces_the_whole_acquisition_frc():
    x, y, t, tid = _structure()
    static = pc.frc_resolution(x, y, tid, repeats=3)
    course = pc.frc_time_course(x, y, t, n_steps=5, repeats=3)
    assert course.times_s.size == 5
    assert course.times_s[-1] == pytest.approx(t.max() - t.min())
    assert course.n_points[-1] == x.size
    assert course.pixel_size_nm == pytest.approx(static.pixel_size_nm)
    assert course.resolution_nm[-1] == pytest.approx(static.resolution_nm)
    assert course.final_resolution_nm == pytest.approx(static.resolution_nm)


def test_resolution_improves_as_the_acquisition_fills_the_structure():
    x, y, t, _tid = _structure(seed=3)
    course = pc.frc_time_course(x, y, t, n_steps=6, repeats=2)
    assert np.all(np.diff(course.n_points) > 0)
    finite = course.resolution_nm[np.isfinite(course.resolution_nm)]
    assert finite.size >= 4
    assert finite[-1] < finite[0]
    assert np.all(np.isfinite(course.spread_nm[np.isfinite(course.resolution_nm)]))


def test_every_window_is_rendered_on_the_same_grid():
    """Same field of view and pixel, so the windows differ only in their points."""
    x, y, t, _tid = _structure(seed=4)
    early = t <= np.percentile(t, 20)
    fov = (x.min(), x.max(), y.min(), y.max())
    own = pc.frc_resolution(x[early], y[early], repeats=1)
    shared = pc.frc_resolution(x[early], y[early], repeats=1, fov_nm=fov)
    full = pc.frc_resolution(x, y, repeats=1)
    assert shared.image_px == full.image_px
    assert shared.pixel_size_nm == pytest.approx(full.pixel_size_nm)
    assert own.freq_inv_nm.size != shared.freq_inv_nm.size or own.pixel_size_nm != pytest.approx(
        shared.pixel_size_nm)


def test_combined_mode_collapses_each_window_to_trace_means():
    x, y, t, tid = _structure(seed=5)
    course = pc.frc_time_course(x, y, t, tid, n_steps=3, mode="combined", repeats=2)
    assert course.mode == "combined"
    # One point per trace present in the window, never more than the localizations.
    assert course.n_points[-1] == np.unique(tid).size
    assert np.all(course.n_points <= np.array([np.sum(t <= t.min() + e) for e in course.times_s]))


def test_progress_reaches_the_total_and_can_stop_the_run():
    x, y, t, _tid = _structure(seed=6, n_emitters=200)
    seen = []
    pc.frc_time_course(x, y, t, n_steps=3, repeats=2,
                       progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (6, 6)

    class Stop(Exception):
        pass

    def stop(done, total):
        if done >= 2:
            raise Stop

    with pytest.raises(Stop):
        pc.frc_time_course(x, y, t, n_steps=3, repeats=2, progress=stop)


def test_per_split_resolutions_are_reported():
    x, y, _t, _tid = _structure(seed=7)
    result = pc.frc_resolution(x, y, repeats=4)
    assert result.repeat_resolutions_nm.shape == (4,)
    assert np.all(np.isfinite(result.repeat_resolutions_nm))


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError):
        pc.frc_time_course(np.zeros(3), np.zeros(3), np.zeros(2))


def test_log_line_is_rendered_by_the_method_text_rules():
    from minflux_viewer.analysis.method_rules import RULES

    x, y, t, _tid = _structure(seed=8, n_emitters=300)
    course = pc.frc_time_course(x, y, t, n_steps=3, repeats=2)
    line = pc.frc_time_course_log_line("run.msr", course)
    for pattern, stage, render in RULES:
        match = pattern.search(line)
        if match:
            sentence, cites = render(match, {"message": line}, None)
            break
    else:
        pytest.fail("no method rule matches the time-course log line")
    assert stage == "analysis"
    assert "3 cumulative time windows" in sentence
    assert "'run.msr'" in sentence
    assert re.search(r"reached \d+\.\d\d nm", sentence)
    assert any("Banterle" in cite[0] for cite in cites)
