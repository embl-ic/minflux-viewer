from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np


def test_alignment_panel_uses_owner_units_persists_steps_and_drags(qtbot):
    from PyQt6.QtCore import QEvent, QPointF, Qt
    from PyQt6.QtGui import QKeyEvent, QMouseEvent
    from PyQt6.QtWidgets import QLabel, QWidget

    from minflux_viewer.ui.overlay_alignment import OverlayAlignmentPanel

    class Owner(QWidget):
        def __init__(self):
            super().__init__()
            self.view = QWidget(self)
            self.saved_steps = []
            self.nudges = []
            self.drags = []

        def _overlay_alignment_control_config(self):
            return {
                "translation_unit": "nm",
                "translation_step": 1.0,
                "translation_maximum": 100000.0,
                "rotation_step": 0.5,
            }

        def _overlay_alignment_steps_changed(self, translation, rotation):
            self.saved_steps.append((translation, rotation))

        def _overlay_alignment_drag_view(self):
            return self.view

        @staticmethod
        def _overlay_alignment_view_delta(start, end):
            return end.x() - start.x(), end.y() - start.y()

        def _overlay_alignment_drag(self, index, dx_nm, dy_nm):
            self.drags.append((index, dx_nm, dy_nm))

        def _overlay_alignment_nudge(self, index, dx_nm, dy_nm, rotation):
            self.nudges.append((index, dx_nm, dy_nm, rotation))

        @staticmethod
        def _overlay_alignment_rotation_sign():
            # Default top-left XY view: mathematical positive appears clockwise.
            return -1.0

        @staticmethod
        def _overlay_alignment_status(_index):
            return "X +0.0 nm | Y +0.0 nm | rotation +0.0°"

        @staticmethod
        def _overlay_alignment_visibility(_index, _visible):
            pass

        @staticmethod
        def _overlay_alignment_set_channel(_index):
            pass

        @staticmethod
        def _overlay_alignment_reset():
            pass

        @staticmethod
        def _overlay_alignment_cancel():
            pass

        @staticmethod
        def _overlay_alignment_apply():
            pass

    owner = Owner()
    panel = OverlayAlignmentPanel(
        owner,
        [{"name": "channel", "visible": True, "lut": "Red"}],
    )
    qtbot.addWidget(owner)
    try:
        labels = [label.text() for label in panel.findChildren(QLabel)]
        assert "nm," in labels
        assert not any(text.startswith("{") and text.endswith("}") for text in labels)
        help_text = panel._help_label.text()
        assert help_text == (
            "mouse drag in the view, or use arrow keys ↔↕ to move horizontally/vertically; "
            "comma ⸴ = ↺ period · = ↻ to rotate"
        )
        assert panel._help_label.textFormat() == Qt.TextFormat.PlainText
        assert "font-size: 12pt" in panel._help_label.styleSheet()
        root_layout = panel.layout()
        help_index = root_layout.indexOf(panel._help_label)
        assert root_layout.itemAt(2).spacerItem().sizeHint().height() >= 8
        assert root_layout.itemAt(help_index - 1).layout() is not None
        assert root_layout.itemAt(help_index + 1).layout() is not None
        assert panel._translation_spin.value() == 1.0
        assert panel._degree_spin.value() == 0.5

        panel._translation_spin.setValue(2.5)
        panel._degree_spin.setValue(0.75)
        assert owner.saved_steps[-1] == (2.5, 0.75)

        key = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Right,
            Qt.KeyboardModifier.NoModifier,
        )
        panel.keyPressEvent(key)
        assert owner.nudges[-1] == (0, 2.5, 0.0, 0.0)

        comma = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Comma,
            Qt.KeyboardModifier.NoModifier,
            ",",
        )
        panel.keyPressEvent(comma)
        assert owner.nudges[-1] == (0, 0.0, 0.0, -0.75)

        period = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Period,
            Qt.KeyboardModifier.NoModifier,
            ".",
        )
        panel.keyPressEvent(period)
        assert owner.nudges[-1] == (0, 0.0, 0.0, 0.75)

        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(10.0, 10.0),
            QPointF(10.0, 10.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(20.0, 25.0),
            QPointF(20.0, 25.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(20.0, 25.0),
            QPointF(20.0, 25.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        assert panel.eventFilter(owner.view, press)
        assert panel.eventFilter(owner.view, move)
        assert panel.eventFilter(owner.view, release)
        assert owner.drags[-1] == (0, 10.0, 15.0)
    finally:
        panel.detach()
        owner.close()


def test_overlay_rotation_sign_tracks_displayed_y_direction():
    from minflux_viewer.ui.render_window import RenderWindow
    from minflux_viewer.ui.scatter_window import ScatterWindow

    render = SimpleNamespace(_should_invert_y_axis=lambda: True)
    assert RenderWindow._overlay_alignment_rotation_sign(render) == -1.0
    render._should_invert_y_axis = lambda: False
    assert RenderWindow._overlay_alignment_rotation_sign(render) == 1.0

    axis = SimpleNamespace(currentText=lambda: "XY")
    scatter = SimpleNamespace(_axis_combo=axis, _xy_origin_top_left=lambda: True)
    assert ScatterWindow._overlay_alignment_rotation_sign(scatter) == -1.0
    scatter._xy_origin_top_left = lambda: False
    assert ScatterWindow._overlay_alignment_rotation_sign(scatter) == 1.0
    axis.currentText = lambda: "XZ"
    scatter._xy_origin_top_left = lambda: True
    assert ScatterWindow._overlay_alignment_rotation_sign(scatter) == 1.0


def test_scatter_alignment_nudge_is_physical_nm_and_steps_persist():
    from minflux_viewer.ui.scatter_window import ScatterWindow

    class State:
        def __init__(self):
            self.prefs = {"plot": {}}
            self.saved = 0

        def save_prefs(self):
            self.saved += 1

    owner = SimpleNamespace(
        _channels=[{"transform": {}}],
        _state=State(),
        _cached_dataset_idx=0,
        _cached_locs_nm=np.ones((1, 3)),
        redraws=0,
    )

    def ensure(_self, channel):
        channel["transform"].setdefault("dx_nm", 0.0)
        channel["transform"].setdefault("dy_nm", 0.0)
        channel["transform"].setdefault("angle", 0.0)

    def redraw(self, *, save_state):
        assert not save_state
        self.redraws += 1

    owner._ensure_channel_world_transform = MethodType(ensure, owner)
    owner._redraw_current = MethodType(redraw, owner)
    owner._update_overlay_alignment_transform = MethodType(
        ScatterWindow._update_overlay_alignment_transform, owner
    )

    ScatterWindow._overlay_alignment_nudge(owner, 0, 1.0, -2.0, 0.5)
    assert owner._channels[0]["transform"] == {
        "dx_nm": 1.0,
        "dy_nm": -2.0,
        "angle": 0.5,
    }
    assert ScatterWindow._overlay_alignment_status(owner, 0) == (
        "X +1.0 nm | Y -2.0 nm | rotation +0.5°"
    )
    config = ScatterWindow._overlay_alignment_control_config(owner)
    assert config["translation_unit"] == "nm"
    assert config["translation_step"] == 1.0
    assert config["rotation_step"] == 0.1

    ScatterWindow._overlay_alignment_steps_changed(owner, 3.25, 0.75)
    assert owner._state.prefs["plot"]["scatter_alignment_translation_nm"] == 3.25
    assert owner._state.prefs["plot"]["scatter_alignment_rotation_deg"] == 0.75
    assert owner._state.saved == 1
    restored = ScatterWindow._overlay_alignment_control_config(owner)
    assert restored["translation_step"] == 3.25
    assert restored["rotation_step"] == 0.75


def test_render_alignment_nudge_is_physical_nm_and_steps_persist():
    from minflux_viewer.ui.render_window import RenderWindow

    class State:
        def __init__(self):
            self.prefs = {"plot": {}}
            self.saved = 0

        def save_prefs(self):
            self.saved += 1

    calls = []
    owner = SimpleNamespace(
        _state=State(),
        _channels=[{"transform": {"dx_nm": 2.0, "dy_nm": -3.0, "angle": 0.1}}],
        _update_overlay_alignment_transform=lambda index, dx, dy, rotation: calls.append(
            (index, dx, dy, rotation)
        ),
    )

    config = RenderWindow._overlay_alignment_control_config(owner)
    assert config["translation_unit"] == "nm"
    assert config["translation_step"] == 1.0
    assert config["rotation_step"] == 0.1

    RenderWindow._overlay_alignment_nudge(owner, 0, 1.0, -2.0, 0.1)
    assert calls == [(0, 1.0, -2.0, 0.1)]
    assert RenderWindow._overlay_alignment_status(owner, 0) == (
        "X +2.0 nm | Y -3.0 nm | rotation +0.1°"
    )

    RenderWindow._overlay_alignment_steps_changed(owner, 4.0, 0.25)
    assert owner._state.prefs["plot"]["render_alignment_translation_nm"] == 4.0
    assert owner._state.prefs["plot"]["render_alignment_rotation_deg"] == 0.25
    assert owner._state.saved == 1


def test_overlay_alignment_preferences_migrate_from_pixel_defaults():
    import copy

    from minflux_viewer.core.app_state import DEFAULT_PREFS, _migrate_prefs

    prefs = copy.deepcopy(DEFAULT_PREFS)
    plot = prefs["plot"]
    plot["render_alignment_translation_px"] = 0.5
    plot.pop("render_alignment_translation_nm")
    plot["render_alignment_rotation_deg"] = 0.5
    plot["scatter_alignment_rotation_deg"] = 0.5

    migrated = _migrate_prefs(prefs)
    plot = migrated["plot"]
    assert "render_alignment_translation_px" not in plot
    assert plot["render_alignment_translation_nm"] == 1.0
    assert plot["render_alignment_rotation_deg"] == 0.1
    assert plot["scatter_alignment_rotation_deg"] == 0.1


def test_render_alignment_preview_uses_bilinear_warp_and_frozen_levels(monkeypatch):
    from minflux_viewer.ui import render_window as module

    captured = {}

    def fake_affine(tile, _matrix, **kwargs):
        captured.update(kwargs)
        return np.asarray(tile, dtype=np.float32).copy()

    monkeypatch.setattr(module, "affine_transform", fake_affine)
    channel = {
        "transform": {"dx_nm": 1.0, "dy_nm": 0.0, "angle": 0.1},
        "levels": None,
        "gamma": 1.0,
    }
    owner = SimpleNamespace(
        _overlay_alignment_panel=object(),
        _overlay_alignment_auto_levels={id(channel): (0.0, 10.0)},
        _last_tile_geometry=(0.0, 8.0, 0.0, 8.0),
        _channels=[channel],
        _manual_levels=None,
    )
    owner._current_render_pixel_size_nm = lambda: 1.0
    tile = np.arange(64, dtype=np.float32).reshape(8, 8)

    module.RenderWindow._transformed_tile(owner, tile, channel)
    assert captured["order"] == 1
    assert captured["prefilter"] is False

    owner._compute_render_auto_levels = lambda _tile: (_ for _ in ()).throw(
        AssertionError("interactive auto levels should be frozen")
    )
    norm = module.RenderWindow._normalized_tile(owner, tile, channel)
    assert norm[0, 0] == 0.0
    assert norm[1, 2] == 1.0


def test_render_alignment_preview_caps_resolution_and_uses_uint8_composite():
    from minflux_viewer.ui.render_window import RenderWindow

    scalar = np.zeros((3, 1200, 800), dtype=np.float32)
    preview = RenderWindow._alignment_preview_scalar(scalar)
    assert preview.shape == (3, 512, 341)
    assert preview.dtype == np.float32

    small = np.zeros((2, 64, 80), dtype=np.float32)
    assert RenderWindow._alignment_preview_scalar(small) is small

    red = np.array([[[200, 0, 0]]], dtype=np.uint8)
    green = np.array([[[0, 100, 0]]], dtype=np.uint8)
    additive = RenderWindow._alignment_preview_rgba([red, green], white_bg=False)
    assert additive.dtype == np.uint8
    assert additive.tolist() == [[[200, 100, 0, 255]]]

    red_ink = np.array([[[255, 128, 128]]], dtype=np.uint8)
    green_ink = np.array([[[128, 255, 128]]], dtype=np.uint8)
    subtractive = RenderWindow._alignment_preview_rgba(
        [red_ink, green_ink], white_bg=True
    )
    assert subtractive.tolist() == [[[128, 128, 64, 255]]]


def test_render_alignment_preview_rebuilds_only_dirty_channel():
    from minflux_viewer.ui.render_window import RenderWindow

    class ImageView:
        def __init__(self):
            self.calls = []

        def setImage(self, image, **kwargs):
            self.calls.append((np.asarray(image), kwargs))

    calls = [0, 0]

    def channel_rgb(index):
        calls[index] += 1
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        rgb[..., index] = 40 + index
        return rgb

    owner = SimpleNamespace(
        _overlay_alignment_panel=object(),
        _overlay_alignment_preview_scalar=np.zeros((2, 4, 5), dtype=np.float32),
        _overlay_alignment_preview_rgb={},
        _overlay_alignment_preview_dirty={0, 1},
        _channels=[{"visible": True}, {"visible": True}],
        _white_bg=False,
        _last_tile_geometry=(10.0, 60.0, 20.0, 60.0),
        _image_view=ImageView(),
        _alignment_preview_channel_rgb=channel_rgb,
        _alignment_preview_rgba=lambda rgb, *, white_bg: (
            RenderWindow._alignment_preview_rgba(rgb, white_bg=white_bg)
        ),
    )

    RenderWindow._render_overlay_alignment_preview(owner)
    assert calls == [1, 1]
    image, kwargs = owner._image_view.calls[-1]
    assert image.dtype == np.uint8
    assert image.shape == (4, 5, 4)
    assert kwargs["scale"] == [10.0, 10.0]

    owner._overlay_alignment_preview_dirty.add(1)
    RenderWindow._render_overlay_alignment_preview(owner)
    assert calls == [1, 2]


def test_render_alignment_preview_coalesces_requests_and_marks_changed_channel():
    from minflux_viewer.ui.render_window import RenderWindow

    class Timer:
        def __init__(self):
            self.active = False
            self.starts = 0

        def isActive(self):
            return self.active

        def start(self):
            self.active = True
            self.starts += 1

    timer = Timer()
    owner = SimpleNamespace(
        _overlay_alignment_panel=object(),
        _overlay_alignment_preview_scalar=np.zeros((2, 8, 8), dtype=np.float32),
        _overlay_alignment_preview_dirty=set(),
        _overlay_alignment_preview_timer=timer,
        _last_scalar_tile=None,
        _channels=[{"transform": {}}, {"transform": {}}],
    )

    RenderWindow._request_overlay_alignment_preview(owner, 0)
    RenderWindow._request_overlay_alignment_preview(owner, 1)
    assert owner._overlay_alignment_preview_dirty == {0, 1}
    assert timer.starts == 1


def test_advanced_render_routes_composition_and_tile_results_behind_preview():
    from minflux_viewer.ui.precision_render_window import PrecisionRenderWindow

    requests = []
    owner = SimpleNamespace(
        _overlay_alignment_panel=object(),
        _request_overlay_alignment_preview=lambda: requests.append(True),
        _last_scalar_tile=np.zeros((1, 2, 2), dtype=np.float32),
    )
    PrecisionRenderWindow._compose_from_cache(owner)
    assert requests == [True]

    cached = []
    result = SimpleNamespace(generation=3, key="tile", array=np.ones((2, 2)))
    owner = SimpleNamespace(
        _precision_scheduler=SimpleNamespace(generation=3),
        _active_tile_generation=3,
        _precision_cache=SimpleNamespace(put=lambda key, array: cached.append((key, array))),
        _overlay_alignment_panel=object(),
        _all_active_tiles_cached=lambda: (_ for _ in ()).throw(
            AssertionError("alignment preview must suppress progressive composition")
        ),
    )
    PrecisionRenderWindow._on_precision_tile_result(owner, result)
    assert cached[0][0] == "tile"


def test_render_alignment_cancel_dispatches_to_viewer_exact_compositor():
    from minflux_viewer.ui.render_window import RenderWindow

    calls = []
    owner = SimpleNamespace(
        _overlay_alignment_original=[{"dx_nm": 1.0}],
        _overlay_alignment_original_visibility=[False],
        _channels=[{"transform": {}, "visible": True}],
        _end_overlay_alignment=lambda: calls.append("end"),
        _compose_from_cache=lambda: calls.append("compose"),
        _schedule_render=lambda: calls.append("schedule"),
    )

    RenderWindow._overlay_alignment_cancel(owner)

    assert owner._channels == [{"transform": {"dx_nm": 1.0}, "visible": False}]
    assert calls == ["end", "compose", "schedule"]
