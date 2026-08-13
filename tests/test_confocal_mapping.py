from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from minflux_viewer.core.attributes import plot_attribute_names
from minflux_viewer.core.confocal_mapping import (
    ConfocalCandidate,
    ConfocalCandidateMatch,
    ConfocalMappingTransform,
    attach_confocal_signal,
    detect_confocal_candidates,
    discover_confocal_candidates,
    geometry_match_error,
    mapping_image,
    sample_confocal_signal,
    transform_pixel_coordinates,
)
from minflux_viewer.core.dataset import AttrStore, DataProp, FileInfo, MinfluxDataset
from minflux_viewer.msr.acquisition_roi import AcquisitionRoi

_OVERLAY_DIR = Path(r"D:\Workspace\Microscopes\MINFLUX\sample data\rotar_project_confocal_overlay")
_OVERLAY_FILES = list(_OVERLAY_DIR.glob("*.msr")) if _OVERLAY_DIR.exists() else []


def _candidate(
    *,
    shape=(4, 5),
    z=False,
    name="Ch2 {12}",
    raw_index=4,
) -> ConfocalCandidate:
    shape = (3, 4, 5) if z else tuple(shape)
    return ConfocalCandidate(
        raw_index=raw_index,
        name=name,
        shape=shape,
        axes="ZYX" if z else "YX",
        dtype="int16",
        x_start_m=0.0,
        y_start_m=0.0,
        x_step_m=1.0,
        y_step_m=1.0,
        z_start_m=0.0 if z else None,
        z_step_m=1.0 if z else None,
        bounds_xy_m=((0.0, 5.0), (0.0, 4.0)),
        matches=(ConfocalCandidateMatch("run", "did-1", 0.0, 0.0),),
    )


def test_candidate_detection_excludes_generated_stacks_and_applies_xy_one_percent():
    roi = AcquisitionRoi("did-1", 0.0, 0.0, 10.0, 20.0)
    base = {
        "sizes": (40, 20),
        "ndim": 2,
        "dtype": "uint16",
        "minflux_type": "",
        "offset": (0.0, 0.0),
        "extent_m": None,
    }
    stacks = [
        {**base, "raw_index": 1, "name": "Ch1 {12}", "length": (10.09, 20.19)},
        {**base, "raw_index": 2, "name": "Ch2 {12}", "length": (10.11, 20.0)},
        {**base, "raw_index": 3, "name": "density", "length": (10.0, 20.0)},
        {
            **base,
            "raw_index": 4,
            "name": "anything",
            "length": (10.0, 20.0),
            "minflux_type": "trace",
        },
    ]

    candidates = detect_confocal_candidates(
        stacks,
        [roi],
        [{"dataset_key": "run", "did": "did-1"}],
    )

    assert [candidate.name for candidate in candidates] == ["Ch1 {12}"]
    np.testing.assert_allclose(candidates[0].matches[0].x_error_fraction, 0.009)
    assert candidates[0].matches[0].y_error_fraction < 0.01


@pytest.mark.skipif(not _OVERLAY_FILES, reason="rotar overlay sample .msr not present")
def test_overlay_sample_detects_only_the_two_matching_confocal_channels():
    from minflux_viewer.msr.acquisition_roi import group_by_dataset, read_acquisition_rois

    path = _OVERLAY_FILES[0]
    selected = [
        {"dataset_key": did, "did": did} for did in group_by_dataset(read_acquisition_rois(path))
    ]
    candidates = discover_confocal_candidates(path, selected)
    assert [candidate.name for candidate in candidates] == ["Ch1 {12}", "Ch2 {12}"]


def test_geometry_match_compares_position_as_well_as_size():
    assert geometry_match_error(((1.0, 11.0), (0.0, 20.0)), (0.0, 0.0, 10.0, 20.0)) == (0.1, 0.0)


def test_projection_uses_float64_and_does_not_overflow_int16():
    volume = np.full((5, 2, 2), 30_000, dtype=np.int16)
    projected = mapping_image(volume, "2D")
    assert projected.dtype == np.float64
    assert np.all(projected == 150_000.0)


def test_2d_nearest_bilinear_and_out_of_bounds_sampling():
    candidate = _candidate()
    image = np.arange(20, dtype=np.float64).reshape(4, 5)

    nearest = sample_confocal_signal(
        image,
        candidate,
        [0.5, 1.5, 0.1, 4.9, -1.0],
        [0.5, 2.5, 0.1, 3.9, 0.5],
        method="nearest neighbour",
    )
    np.testing.assert_allclose(nearest[:4], [0.0, 11.0, 0.0, 19.0])
    assert np.isnan(nearest[4])

    bilinear = sample_confocal_signal(
        image,
        candidate,
        [1.0],
        [1.0],
        method="bilinear",
    )
    np.testing.assert_allclose(bilinear, [3.0])


def test_3d_trilinear_sampling_uses_calibrated_z():
    candidate = _candidate(z=True)
    z, y, x = np.indices(candidate.shape)
    volume = z * 100.0 + y * 10.0 + x
    sampled = sample_confocal_signal(
        volume,
        candidate,
        [0.5, 1.5],
        [0.5, 2.5],
        [1.5, 2.5],
        dimension="3D",
        method="trilinear",
    )
    np.testing.assert_allclose(sampled, [100.0, 221.0])


def test_positive_rotation_is_visual_counter_clockwise_about_image_centre():
    x, y = transform_pixel_coordinates(
        [4.0],
        [2.0],
        (5, 5),
        ConfocalMappingTransform(rotation_deg=90.0),
    )
    np.testing.assert_allclose(x, [2.0], atol=1e-12)
    np.testing.assert_allclose(y, [0.0], atol=1e-12)


def test_attached_signal_is_user_visible_and_available_in_raw_store():
    attrs = AttrStore(
        {
            "loc_x": np.array([0.5, 1.5]),
            "loc_y": np.array([0.5, 1.5]),
            "loc_z": np.zeros(2),
            "tid": np.array([1, 2]),
        }
    )
    dataset = MinfluxDataset(
        FileInfo("run", ""),
        prop=DataProp(num_loc=2, num_itr=1, num_dim=2, num_traces=2, attr_names=attrs.keys()),
        attr=attrs,
    )
    dataset.components.mfx_raw = AttrStore(
        {
            "loc_x": np.array([0.5, 1.5]),
            "loc_y": np.array([0.5, 1.5]),
            "loc_z": np.zeros(2),
            "tid": np.array([1, 2]),
            "itr": np.zeros(2, dtype=np.int16),
            "vld": np.ones(2, dtype=bool),
        }
    )
    image = np.arange(20, dtype=np.int16).reshape(4, 5)

    result = attach_confocal_signal(
        dataset,
        "source.msr",
        _candidate(),
        "Ch2",
        method="nearest neighbour",
        image=image,
    )

    np.testing.assert_allclose(dataset.attr["Ch2"], [0.0, 6.0])
    np.testing.assert_allclose(dataset.mfx_raw["Ch2"], [0.0, 6.0])
    assert result.finite_count == 2
    assert dataset.mfx.get_meta("Ch2")["user_visible"] is True
    assert "Ch2" in plot_attribute_names(dataset, {"attributes": {}})
    assert dataset.metadata["confocal_signal_mappings"]["Ch2"]["stack_name"] == "Ch2 {12}"


def test_mapping_options_switch_to_dimension_appropriate_methods(qtbot):
    from minflux_viewer.ui.confocal_mapping_dialog import ConfocalMappingOptionsWidget

    widget = ConfocalMappingOptionsWidget([_candidate(z=True)])
    qtbot.addWidget(widget)
    check, edit, _candidate_row = widget._rows[0]
    check.setChecked(True)
    assert edit.text() == "Ch2"
    widget.dimension_combo.setCurrentText("3D")
    assert [widget.method_combo.itemText(i) for i in range(widget.method_combo.count())] == [
        "nearest neighbour",
        "trilinear",
    ]
    assert widget.options().choices[0].attribute_name == "Ch2"


def test_single_dataset_reader_dialog_shows_confocal_section_not_overlay_controls(qtbot):
    from minflux_viewer.plugins.msr_reader.msr_reader_dialog import ViewerAlignmentDialog

    dialog = ViewerAlignmentDialog(
        ["run"],
        lambda _name: (False, "not applicable"),
        confocal_candidates=[_candidate(z=True)],
    )
    qtbot.addWidget(dialog)
    assert dialog.confocal_widget is not None
    assert dialog.individual_check.isHidden()
    assert dialog.ref_combo.isHidden()
    assert dialog.align_combo.isHidden()


def test_manual_alignment_dialog_uses_editable_steps_and_shortcuts(qtbot):
    from minflux_viewer.ui.confocal_mapping_dialog import ConfocalManualAlignmentDialog

    dialog = ConfocalManualAlignmentDialog(
        _candidate(),
        np.arange(20, dtype=float).reshape(4, 5),
        [0.5, 1.5],
        [0.5, 1.5],
    )
    qtbot.addWidget(dialog)
    assert dialog.windowTitle().startswith("Manual fluorescent channel alignment —")
    dialog._nudge(0.5, -0.5, 0.1)
    assert dialog.transform == ConfocalMappingTransform(0.5, -0.5, 0.1)
    assert "X +0.5 px" in dialog._status.text()
    assert "rotation +0.1°" in dialog._status.text()
    assert dialog._translation_step_spin.value() == pytest.approx(0.5)
    assert dialog._rotation_step_spin.value() == pytest.approx(0.1)
    assert dialog._help_label.text() == (
        "mouse drag in the view, or use arrow keys ↔↕ to move horizontally/vertically; "
        "comma ⸴ = ↺ period · = ↻ to rotate"
    )
    shortcuts = {shortcut.key().toString(): shortcut for shortcut in dialog._shortcuts}
    assert set((",", ".")).issubset(shortcuts)
    assert shortcuts[","].key().toString() == ","
    assert shortcuts["."].key().toString() == "."
    dialog._reset()
    shortcuts[","].activated.emit()
    assert dialog.transform.rotation_deg == pytest.approx(0.1)
    shortcuts["."].activated.emit()
    assert dialog.transform.rotation_deg == pytest.approx(0.0)
    dialog._translation_step_spin.setValue(1.25)
    dialog._rotation_step_spin.setValue(2.0)
    shortcuts["Right"].activated.emit()
    shortcuts["."].activated.emit()
    assert dialog.transform.dx_pixels == pytest.approx(1.25)
    assert dialog.transform.rotation_deg == pytest.approx(-2.0)


def test_manual_alignment_dialog_toggles_shared_channel_preview(qtbot):
    from minflux_viewer.ui.confocal_mapping_dialog import ConfocalManualAlignmentDialog

    ch1 = _candidate(name="Ch1 {12}", raw_index=3)
    ch2 = _candidate(name="Ch2 {12}", raw_index=4)
    image1 = np.arange(20, dtype=float).reshape(4, 5)
    image2 = np.fliplr(image1)
    dialog = ConfocalManualAlignmentDialog(
        ch1,
        image1,
        [0.5, 1.5],
        [0.5, 1.5],
        channels=[(ch1, image1), (ch2, image2)],
    )
    qtbot.addWidget(dialog)

    assert [check.text() for check in dialog._image_checks] == ["Ch1 {12}", "Ch2 {12}"]
    assert all(check.isChecked() for check in dialog._image_checks)
    assert dialog._image_item.image.shape == (4, 5, 3)
    assert dialog._localization_check.text() == "mfx.loc"
    assert dialog._localization_color_combo.currentText() == "Cyan"
    assert [dialog._localization_color_combo.itemText(i)
            for i in range(dialog._localization_color_combo.count())] == [
        "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Orange",
        "White", "Gray", "Black",
    ]
    assert [combo.currentText() for combo in dialog._image_color_combos] == [
        "Green",
        "Magenta",
    ]
    assert dialog._image_color_combos[0].minimumWidth() == 101
    assert dialog._translation_step_spin.minimumWidth() == 108
    assert dialog._rotation_step_spin.minimumWidth() == 108

    dialog._image_checks[1].setChecked(False)
    assert dialog._image_item.image.shape == (4, 5, 3)
    np.testing.assert_allclose(
        dialog._image_item.image[..., 1], dialog._normalized_images[0]
    )
    np.testing.assert_allclose(dialog._image_item.image[..., (0, 2)], 0.0)

    root = dialog.layout()
    channel_row = next(
        index
        for index in range(root.count())
        if root.itemAt(index).layout() is dialog._channel_controls
    )
    report_row = next(
        index
        for index in range(root.count())
        if root.itemAt(index).layout() is dialog._transform_report_row
    )
    help_row = root.indexOf(dialog._help_label)
    assert channel_row < help_row < report_row
    assert dialog._channel_controls.indexOf(dialog._reset_button) > 0


def test_manual_alignment_steps_persist_via_owner_state(qtbot):
    from PyQt6.QtWidgets import QWidget

    from minflux_viewer.ui.confocal_mapping_dialog import ConfocalManualAlignmentDialog

    class State:
        def __init__(self):
            self.prefs = {"plot": {}}
            self.saved = 0

        def save_prefs(self):
            self.saved += 1

    parent = QWidget()
    parent._state = State()
    qtbot.addWidget(parent)
    dialog = ConfocalManualAlignmentDialog(
        _candidate(),
        np.arange(20, dtype=float).reshape(4, 5),
        [0.5, 1.5],
        [0.5, 1.5],
        parent,
    )
    qtbot.addWidget(dialog)

    dialog._translation_step_spin.setValue(0.75)
    dialog._rotation_step_spin.setValue(0.2)
    plot = parent._state.prefs["plot"]
    assert plot["confocal_alignment_translation_px"] == pytest.approx(0.75)
    assert plot["confocal_alignment_rotation_deg"] == pytest.approx(0.2)
    assert parent._state.saved == 2


def test_manual_mapping_groups_same_geometry_channels_into_one_dialog(monkeypatch):
    from types import SimpleNamespace

    from PyQt6.QtWidgets import QDialog

    from minflux_viewer.ui import confocal_mapping_dialog as module

    ch1 = _candidate(name="Ch1 {12}", raw_index=3)
    ch2 = _candidate(name="Ch2 {12}", raw_index=4)
    options = module.ConfocalMappingOptions(
        choices=(
            module.ConfocalMappingChoice(ch1, "Ch1"),
            module.ConfocalMappingChoice(ch2, "Ch2"),
        ),
        dimension="2D",
        method="bilinear",
        alignment="manual",
    )
    dataset = SimpleNamespace(name="run", metadata={"msr_dataset_key": "run"})
    opened = []
    attached = []
    shared = ConfocalMappingTransform(1.0, -0.5, 0.2)

    class FakeDialog:
        def __init__(self, _candidate_arg, _image, _x, _y, _parent, **kwargs):
            opened.append([candidate.name for candidate, _image in kwargs["channels"]])
            self.transform = shared

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(module, "ConfocalManualAlignmentDialog", FakeDialog)
    monkeypatch.setattr(
        module,
        "load_confocal_candidate_array",
        lambda _path, candidate: np.full(candidate.shape, candidate.raw_index, dtype=float),
    )
    monkeypatch.setattr(
        "minflux_viewer.core.loader.attr_values_1d",
        lambda _dataset, _name: np.array([0.5, 1.5]),
    )

    def fake_attach(_dataset, _path, candidate, attribute_name, **kwargs):
        attached.append((candidate.name, attribute_name, kwargs["transform"]))
        return attribute_name

    monkeypatch.setattr(module, "attach_confocal_signal", fake_attach)

    results = module.apply_confocal_mapping_options(dataset, "source.msr", options)

    assert opened == [["Ch1 {12}", "Ch2 {12}"]]
    assert results == ["Ch1", "Ch2"]
    assert attached == [
        ("Ch1 {12}", "Ch1", shared),
        ("Ch2 {12}", "Ch2", shared),
    ]
