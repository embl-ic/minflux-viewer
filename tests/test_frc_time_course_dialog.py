"""The FRC window's *Resolution over acquisition time* section."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _points(seed: int = 0, n_emitters: int = 400, per: int = 6):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0.0, 1500.0, size=(n_emitters, 2))
    idx = np.repeat(np.arange(n_emitters), per)
    xy = centres[idx] + rng.normal(0.0, 4.0, size=(idx.size, 2))
    t = rng.uniform(0.0, 300.0, n_emitters)[idx]
    return xy[:, 0], xy[:, 1], t, idx


class _State:
    def __init__(self):
        self.lines = []

    def log(self, text, level="INFO", **_kw):
        self.lines.append(text)


def _delete_now(_app, widget):
    from PyQt6.QtCore import QEvent

    widget.close()
    widget.deleteLater()
    _app.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def test_time_course_runs_in_the_background_and_is_plotted(qapp, qtbot):
    from minflux_viewer.analysis import precision_calculation as pc
    from minflux_viewer.ui.analysis.precision_analysis import FRCResultDialog

    x, y, t, tid = _points()
    result = pc.frc_resolution(x, y, tid, repeats=2)
    state = _State()
    dialog = FRCResultDialog(result, x, y, tid, t_s=t, state=state, dataset_name="run.msr")
    try:
        assert dialog.windowTitle() == "Resolution — FRC"
        assert dialog._course_btn.isEnabled()
        dialog._course_steps.setValue(3)
        dialog._course_splits.setValue(2)
        dialog._start_time_course()
        assert not dialog._course_btn.isEnabled()
        qtbot.waitUntil(lambda: dialog._course is not None and dialog._course_task is None,
                        timeout=60000)
        course = dialog._course
        assert course.times_s.size == 3
        assert course.final_resolution_nm == pytest.approx(result.resolution_nm)
        assert any(line.startswith("Resolution over acquisition time (FRC) of 'run.msr'")
                   for line in state.lines)
        assert dialog._course_btn.isEnabled()
    finally:
        _delete_now(qapp, dialog)


def test_without_time_stamps_the_section_says_why(qapp):
    from minflux_viewer.analysis import precision_calculation as pc
    from minflux_viewer.ui.analysis.precision_analysis import FRCResultDialog

    x, y, _t, _tid = _points(seed=1, n_emitters=150)
    result = pc.frc_resolution(x, y, repeats=1)
    dialog = FRCResultDialog(result, x, y)
    try:
        assert not dialog._course_btn.isEnabled()
        assert "no time stamps" in dialog._course_status.text()
    finally:
        _delete_now(qapp, dialog)
