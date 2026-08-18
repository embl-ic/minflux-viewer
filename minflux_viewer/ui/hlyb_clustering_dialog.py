"""Parameter dialog + result window for the HlyB sub-unit pair analysis
(Analyze › Clustering › HlyB subunit pair analysis).

``HlyBClusteringDialog`` is a small modal parameter picker; ``HlyBResultWindow``
is a modeless result window with a 3-D scatter of the detected sub-units / HlyB
structures / measured pair links and a histogram of all pair distances.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPoint, QTimer, Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ..analysis.hlyb_clustering import HlyBConfig

# Cap distance labels so a busy 2-D or 3-D field of view stays legible.
_MAX_DISTANCE_LABELS = 100
_MAX_RAW_POINTS = 100_000
_PAIR_LABEL_MIN_PIXELS = 42.0
_HISTOGRAM_MIN_MAX_NM = 40.0
_DISTANCE_ZOOM_VALUES_PER_BIN = 4
_DISTANCE_ZOOM_MIN_BINS = 10
_DISTANCE_ZOOM_MAX_BINS = 60
_DISTANCE_ZOOM_Y_HEADROOM = 0.05

_DISTANCE_ZOOM_TOOLTIPS = {
    "horizontal": (
        "Drag a horizontal guide at the mouse; X remaps to the drawn span.\n"
        "The bin size is refined for the new span and the height is\n"
        "re-fitted to the re-binned bars."
    ),
    "vertical": (
        "Drag a vertical guide at the mouse; Y (count) remaps to the drawn\n"
        "span, clamped at zero. X and the bin size are unchanged."
    ),
    "unconstrained": (
        "Drag a rectangle; X remaps to it and the bin size is refined.\n"
        "The height is re-fitted to the re-binned bars."
    ),
}


def _raw_color_attributes_for_result(
    result: dict,
    source_dataset=None,
    prefs: dict | None = None,
) -> dict[str, np.ndarray]:
    """Return source-dataset attributes aligned to result raw-localization rows."""
    n_points = int(np.asarray(result.get("points_nm", np.empty((0, 3)))).shape[0])
    attributes: dict[str, np.ndarray] = {}
    for name, values in (result.get("raw_color_attributes", {}) or {}).items():
        arr = np.asarray(values)
        if arr.ndim == 1 and arr.size == n_points and arr.dtype.kind in "biufc":
            attributes[str(name)] = arr.astype(float, copy=False)
    if source_dataset is None:
        return attributes

    from ..core.attributes import plot_attribute_names
    from ..core.loader import (
        attr_values_1d,
        effective_iteration_for_attr,
        iteration_attr_values_1d,
        mfx_get,
    )

    source_x = mfx_get(source_dataset, "loc_x", itr="last", vld_only=True)
    source_x = np.empty(0) if source_x is None else np.asarray(source_x).ravel()
    source_count = int(source_x.size)
    source_indices = np.asarray(
        result.get("point_source_indices", np.arange(n_points)), dtype=np.int64,
    ).ravel()
    if (source_indices.size != n_points or source_count == 0
            or np.any(source_indices < 0) or np.any(source_indices >= source_count)):
        return attributes

    names = plot_attribute_names(
        source_dataset, prefs or {}, exclude=("ftr", "idx"),
    )
    # Trace identity is essential for this analysis and remains available even
    # when a custom Preferences attribute list accidentally omits it.
    if "tid" not in names:
        names.insert(0, "tid")

    for name in names:
        effective_iteration = effective_iteration_for_attr(source_dataset, name)
        if effective_iteration is not None:
            values = iteration_attr_values_1d(
                source_dataset, name, effective_iteration,
                itr="last", vld_only=True,
            )
        else:
            values = mfx_get(
                source_dataset, name, itr="last", vld_only=True,
            )
        arr = np.empty(0) if values is None else np.asarray(values)
        if arr.ndim != 1 or arr.size != source_count:
            materialized = attr_values_1d(source_dataset, name)
            arr = np.empty(0) if materialized is None else np.asarray(materialized)
        if (arr.ndim == 1 and arr.size == source_count
                and arr.dtype.kind in "biufc"):
            attributes[name] = arr[source_indices].astype(float, copy=False)
    return attributes


def _adaptive_histogram_upper_nm(values: np.ndarray, width: float) -> float:
    """Rounded histogram upper bound with a 40 nm minimum and small headroom."""
    data_max = float(np.max(values))
    if data_max <= _HISTOGRAM_MIN_MAX_NM:
        return _HISTOGRAM_MIN_MAX_NM
    target = data_max + max(float(width), 0.05 * data_max)
    if target <= 100.0:
        quantum = 5.0
    elif target <= 1000.0:
        quantum = 10.0
    elif target <= 5000.0:
        quantum = 50.0
    else:
        magnitude = 10.0 ** np.floor(np.log10(target))
        quantum = magnitude / 10.0
    return float(np.ceil(target / quantum) * quantum)


class _HoverHistogramBarItem(pg.BarGraphItem):
    """Histogram bars with a bin-centre/count tooltip on mouse hover."""

    def __init__(
        self,
        *,
        counts: np.ndarray,
        edges: np.ndarray,
        series_name: str,
        **opts,
    ) -> None:
        self._counts = np.asarray(counts, dtype=int).ravel()
        self._edges = np.asarray(edges, dtype=float).ravel()
        self._series_name = str(series_name)
        super().__init__(**opts)
        self.setAcceptHoverEvents(True)

    def tooltip_for_x(self, x: float) -> str | None:
        if self._edges.size != self._counts.size + 1:
            return None
        idx = int(np.searchsorted(self._edges, float(x), side="right") - 1)
        if not (0 <= idx < self._counts.size):
            return None
        center = 0.5 * (self._edges[idx] + self._edges[idx + 1])
        return (
            f"{self._series_name}\n"
            f"Bin center: {center:.2f} nm\n"
            f"Count: {int(self._counts[idx]):,}"
        )

    def hoverEvent(self, event) -> None:  # noqa: N802 - Qt event-handler name
        if event.isExit():
            QToolTip.hideText()
            return
        tooltip = self.tooltip_for_x(float(event.pos().x()))
        if tooltip is None:
            QToolTip.hideText()
            return
        screen = event.screenPos()
        QToolTip.showText(QPoint(round(screen.x()), round(screen.y())), tooltip)


class HlyBClusteringDialog(QDialog):
    """Modal parameter picker for the HlyB sub-unit pair analysis."""

    def __init__(self, parent=None, *, defaults: HlyBConfig | None = None, mode: str = "3D") -> None:
        super().__init__(parent)
        mode_text = str(mode).upper()
        if "TEMPLATE" in mode_text:
            self._mode = "TEMPLATE2D" if "2D" in mode_text else "TEMPLATE3D"
            mode_label = f"Template matching {self._mode[-2:]}"
        else:
            self._mode = "2D" if mode_text == "2D" else "3D"
            mode_label = self._mode
        self.setWindowTitle(f"HlyB Subunit Pair Analysis ({mode_label})")
        self.setModal(True)
        d = defaults or HlyBConfig()

        root = QVBoxLayout(self)
        intro_text = (
            "Detect protein sub-units from MINFLUX traces, cluster them into HlyB "
            "structures, and measure the sub-unit pair distances within each structure."
        )
        if self._mode in {"2D", "TEMPLATE2D"}:
            intro_text += (
                " Each E.coli is delineated from the localization density and shrunk "
                "inward; traces in that border margin are excluded, because there the "
                "membrane is seen edge-on and an in-plane distance is systematically "
                "foreshortened. Note that the 2-D projection still superimposes the "
                "upper and lower membrane, which this step does not correct."
            )
        if self._mode in {"TEMPLATE2D", "TEMPLATE3D"}:
            intro_text += (
                " Candidate sub-units are assigned to a partial six-site, C3-symmetric "
                "HlyB core. Pair distances are derived from one consistent geometry; "
                "the sdAb displacement is treated as matching uncertainty."
            )
        if self._mode == "TEMPLATE2D":
            intro_text += (
                " Distances are measured in the image plane, where a tilted complex "
                "appears shorter but never longer. The matcher therefore forgives "
                "shortening up to the tilt the border shrink admits, so a genuine "
                "complex is not rejected for being tilted."
            )
        intro = QLabel(intro_text)
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self._min_loc = QSpinBox()
        self._min_loc.setRange(1, 100000)
        self._min_loc.setValue(int(d.min_loc_per_trace))
        form.addRow("Min loc per trace:", self._min_loc)

        self._zscale = None
        self._border = None
        self._mask_px = None
        self._border_mode = None
        self._border_fraction = None
        self._mask_smooth = None
        self._mask_close = None
        if self._mode in {"3D", "TEMPLATE3D"}:
            self._zscale = self._dspin(0.1, 2.0, d.z_scaling_factor, 4, 0.01, "")
            self._zscale.setToolTip(
                "Factor applied to the raw z coordinate before analysis "
                "(z_nm = raw_z × this). Defaults to the dataset's current RIMF; "
                "edit it here to override for this run.")
            form.addRow("Z scaling (RIMF):", self._zscale)
        else:
            self._border_mode = QComboBox()
            self._border_mode.addItem("absolute (nm)", "absolute")
            self._border_mode.addItem("relative (fraction of cell half-width)", "relative")
            self._border_mode.setCurrentIndex(
                1 if str(d.border_mode).lower() == "relative" else 0)
            self._border_mode.setToolTip(
                "How far to shrink each cell inward.\n"
                "absolute — a fixed distance in nm (the original MATLAB behaviour).\n"
                "relative — a fraction of that cell's own half-width, so wide and\n"
                "narrow cells are treated alike. Relative also bounds the membrane\n"
                "tilt directly: keeping a fraction f of the half-width admits only\n"
                "surface normals within arcsin(1 - f) of face-on, which is exactly\n"
                "the foreshortening the shrink is meant to avoid.")
            form.addRow("Border shrink mode:", self._border_mode)

            self._border = self._dspin(0.0, 2000.0, d.border_size_nm, 1, 10.0, " nm")
            self._border.setToolTip("Absolute mode: shrink each cell mask inward by "
                                    "this distance; traces in the margin are excluded.")
            form.addRow("Border shrink (absolute):", self._border)

            self._border_fraction = self._dspin(0.0, 0.95, d.border_fraction, 2, 0.05, "")
            self._border_fraction.setToolTip(
                "Relative mode: drop localizations within this fraction of the cell "
                "half-width from the boundary. 0.35 admits normals within ~40° of "
                "face-on; larger values are stricter but keep less data.")
            form.addRow("Border shrink (relative):", self._border_fraction)

            self._mask_px = self._dspin(1.0, 200.0, d.mask_pixel_size_nm, 1, 1.0, " nm")
            self._mask_px.setToolTip("Pixel size of the coarse histogram used to build the cell mask.")
            form.addRow("Mask pixel size:", self._mask_px)

            self._mask_smooth = self._dspin(0.0, 500.0, d.mask_smooth_nm, 0, 10.0, " nm")
            self._mask_smooth.setToolTip(
                "Gaussian smoothing radius applied to the density before thresholding. "
                "Must exceed the typical gap between labelled puncta, or the cell "
                "fragments into blobs instead of forming one body.")
            form.addRow("Mask smoothing:", self._mask_smooth)

            self._mask_close = self._dspin(0.0, 1000.0, d.mask_close_nm, 0, 10.0, " nm")
            self._mask_close.setToolTip(
                "Morphological closing radius, applied after thresholding to bridge "
                "remaining gaps between puncta without shrinking the cell outline.")
            form.addRow("Mask closing:", self._mask_close)

            def _sync_border_mode() -> None:
                relative = self._border_mode.currentData() == "relative"
                self._border.setEnabled(not relative)
                self._border_fraction.setEnabled(relative)

            self._border_mode.currentIndexChanged.connect(lambda _i: _sync_border_mode())
            _sync_border_mode()

        self._pixel = self._dspin(0.25, 50.0, d.unit_render_pixel_size, 2, 0.5, " nm")
        form.addRow("Unit render pixel size:", self._pixel)

        self._unit_size = self._dspin(0.0, 500.0, d.basic_unit_size_nm, 2, 1.0, " nm")
        self._unit_size.setSpecialValueText("auto")
        self._unit_size.setToolTip("Basic sub-unit diameter (Dunit). 0 = auto from trace size.")
        form.addRow("Basic unit size:", self._unit_size)

        self._min_units = QSpinBox()
        if self._mode in {"TEMPLATE2D", "TEMPLATE3D"}:
            self._min_units.setRange(2, 6)
            self._min_units.setValue(int(d.min_observed_subunits_per_HlyB))
            self._min_units.setToolTip(
                "Minimum observed sub-units for an accepted partial HlyB match. "
                "Three is the default because a two-point match is structurally ambiguous."
            )
            form.addRow("Min observed sub-units:", self._min_units)
        else:
            self._min_units.setRange(1, 100)
            self._min_units.setValue(int(d.min_unit_count_per_HlyB))
            self._min_units.setToolTip("DBSCAN minPts: smallest number of sub-units forming an HlyB.")
            form.addRow("Min units per HlyB:", self._min_units)

        self._d1a1b = self._dspin(1.0, 200.0, d.diameter_distance_1a1b_nm, 1, 1.0, " nm")
        self._d1a2a = self._dspin(1.0, 200.0, d.diameter_distance_1a2a_nm, 1, 1.0, " nm")
        self._d1b2b = self._dspin(1.0, 200.0, d.diameter_distance_1b2b_nm, 1, 1.0, " nm")
        self._d1b2b.setToolTip("Sets the HlyB clustering radius (2·d/√3).")
        if self._mode not in {"TEMPLATE2D", "TEMPLATE3D"}:
            form.addRow("Distance 1a–1b (prior):", self._d1a1b)
            form.addRow("Distance 1a–2a (prior):", self._d1a2a)
            form.addRow("Distance 1b–2b (prior):", self._d1b2b)

        self._neighbor = None
        self._cross = None
        self._core_a_side = None
        self._core_b_side = None
        self._core_twist = None
        self._core_axial = None
        self._label_offset = None
        self._template_tol = None
        self._template_rms = None
        self._template_max_resid = None
        self._template_match_frac = None
        if self._mode in {"TEMPLATE2D", "TEMPLATE3D"}:
            self._core_a_side = self._dspin(
                1.0, 200.0, d.template_core_a_ring_side_nm, 1, 0.1, " nm")
            self._core_b_side = self._dspin(
                1.0, 200.0, d.template_core_b_ring_side_nm, 1, 0.1, " nm")
            self._core_twist = self._dspin(
                -180.0, 180.0, d.template_core_twist_deg, 2, 0.5, "°")
            self._core_axial = self._dspin(
                0.0, 100.0, d.template_core_axial_offset_nm, 1, 0.5, " nm")
            self._core_axial.setSpecialValueText("coplanar")
            self._label_offset = self._dspin(
                0.0, 20.0, d.template_label_offset_nm, 1, 0.1, " nm/site")
            self._label_offset.setToolTip(
                "Approximate displacement of each fluorophore from its HlyB domain. "
                "It informs the automatic tolerance and is not added to every pair distance."
            )
            form.addRow("Core A-ring side:", self._core_a_side)
            form.addRow("Core B-ring side:", self._core_b_side)
            form.addRow("B-ring twist:", self._core_twist)
            form.addRow("A/B axial offset:", self._core_axial)
            form.addRow("Label offset uncertainty:", self._label_offset)

            self._template_tol = self._dspin(0.0, 50.0, d.model_pair_tolerance_nm, 1, 0.5, " nm")
            self._template_tol.setSpecialValueText("auto")
            self._template_tol.setToolTip(
                "Pair-distance residual tolerance. 0 = two label offsets plus 1 nm; "
                "unlike the former behavior, this is independent of Dunit."
            )
            form.addRow("Template pair tolerance:", self._template_tol)

            self._template_rms = self._dspin(0.0, 50.0, d.model_rms_threshold_nm, 1, 0.5, " nm")
            self._template_rms.setSpecialValueText("auto")
            self._template_rms.setToolTip("Accepted-template RMS residual. 0 = 0.8 × pair tolerance.")
            form.addRow("Template RMS threshold:", self._template_rms)

            self._template_max_resid = self._dspin(0.0, 80.0, d.model_max_residual_nm, 1, 0.5, " nm")
            self._template_max_resid.setSpecialValueText("auto")
            self._template_max_resid.setToolTip("Maximum single-pair residual. 0 = 1.6 × pair tolerance.")
            form.addRow("Max pair residual:", self._template_max_resid)

            self._template_match_frac = self._dspin(0.0, 1.0, d.min_pair_match_fraction, 2, 0.05, "")
            self._template_match_frac.setToolTip("Fraction of pairs that must be within tolerance.")
            form.addRow("Min match fraction:", self._template_match_frac)
        root.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Analyze")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    @staticmethod
    def _dspin(lo, hi, val, decimals, step, suffix) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(float(val))
        if suffix:
            s.setSuffix(suffix)
        return s

    def config(self) -> HlyBConfig:
        defaults = HlyBConfig()
        return HlyBConfig(
            min_loc_per_trace=int(self._min_loc.value()),
            z_scaling_factor=(float(self._zscale.value()) if self._zscale is not None else 1.0),
            unit_render_pixel_size=float(self._pixel.value()),
            basic_unit_size_nm=float(self._unit_size.value()),
            min_unit_count_per_HlyB=int(self._min_units.value()),
            diameter_distance_1a1b_nm=float(self._d1a1b.value()),
            diameter_distance_1a2a_nm=float(self._d1a2a.value()),
            diameter_distance_1b2b_nm=float(self._d1b2b.value()),
            neighboring_domain_distance_nm=(
                float(self._neighbor.value()) if self._neighbor is not None
                else defaults.neighboring_domain_distance_nm
            ),
            cross_domain_distance_nm=(
                float(self._cross.value()) if self._cross is not None
                else defaults.cross_domain_distance_nm
            ),
            template_core_a_ring_side_nm=(
                float(self._core_a_side.value()) if self._core_a_side is not None
                else defaults.template_core_a_ring_side_nm
            ),
            template_core_b_ring_side_nm=(
                float(self._core_b_side.value()) if self._core_b_side is not None
                else defaults.template_core_b_ring_side_nm
            ),
            template_core_twist_deg=(
                float(self._core_twist.value()) if self._core_twist is not None
                else defaults.template_core_twist_deg
            ),
            template_core_axial_offset_nm=(
                float(self._core_axial.value()) if self._core_axial is not None
                else defaults.template_core_axial_offset_nm
            ),
            template_label_offset_nm=(
                float(self._label_offset.value()) if self._label_offset is not None
                else defaults.template_label_offset_nm
            ),
            min_observed_subunits_per_HlyB=(
                int(self._min_units.value())
                if self._mode in {"TEMPLATE2D", "TEMPLATE3D"}
                else defaults.min_observed_subunits_per_HlyB
            ),
            model_pair_tolerance_nm=(
                float(self._template_tol.value()) if self._template_tol is not None
                else defaults.model_pair_tolerance_nm
            ),
            model_rms_threshold_nm=(
                float(self._template_rms.value()) if self._template_rms is not None
                else defaults.model_rms_threshold_nm
            ),
            model_max_residual_nm=(
                float(self._template_max_resid.value()) if self._template_max_resid is not None
                else defaults.model_max_residual_nm
            ),
            min_pair_match_fraction=(
                float(self._template_match_frac.value()) if self._template_match_frac is not None
                else defaults.min_pair_match_fraction
            ),
            border_size_nm=(float(self._border.value()) if self._border is not None else defaults.border_size_nm),
            border_mode=(
                str(self._border_mode.currentData()) if self._border_mode is not None
                else defaults.border_mode
            ),
            border_fraction=(
                float(self._border_fraction.value()) if self._border_fraction is not None
                else defaults.border_fraction
            ),
            mask_pixel_size_nm=(float(self._mask_px.value()) if self._mask_px is not None else defaults.mask_pixel_size_nm),
            mask_smooth_nm=(
                float(self._mask_smooth.value()) if self._mask_smooth is not None
                else defaults.mask_smooth_nm
            ),
            mask_close_nm=(
                float(self._mask_close.value()) if self._mask_close is not None
                else defaults.mask_close_nm
            ),
            # The legacy pixel-indexed smoothing override must not leak back in:
            # the dialog now sets the smoothing radius in nanometres.
            mask_sigma_px=0.0,
        )


class HlyBResultWindow(QDialog):
    """Modeless result window: 3-D sub-unit/pair scatter + pair-distance histogram."""

    def __init__(
        self,
        result: dict,
        cfg: HlyBConfig,
        *,
        title: str = "",
        owner=None,
        prefer_2d: bool | None = None,
        source_dataset=None,
        prefs: dict | None = None,
    ) -> None:
        super().__init__(None)
        self._owner = owner
        self._result = result
        self._cfg = cfg
        self._prefs = prefs or {}
        self._distance_hist_plot = None
        self._distance_bin_spin = None
        self._distance_stats_label = None
        self._show_lognormal_fit_checkbox = None
        self._fit_shape_combo = None
        self._show_all_pairs_checkbox = None
        self._distance_bar_item = None
        self._all_distance_bar_item = None
        self._all_pair_histogram_cache = {}
        self._all_pair_histogram_error = None
        self._lognormal_fit_requested = False
        self._lognormal_fit_cache = {}
        self._lognormal_fit_result = None
        self._lognormal_curve_item = None
        self._lognormal_mean_line = None
        self._lognormal_fit_text = None
        self._distance_zoom_mode = None
        self._distance_zoom_drag_start = None
        self._distance_zoom_preview = None
        self._distance_original_mouse_drag_event = None
        self._distance_view_box = None
        self._distance_reset_bin_size = None
        self._distance_zoom_rebinning = False
        self._pair_distances_nm = np.asarray(
            result.get("all_pair_distances", np.empty(0)), dtype=float).ravel()
        self._pair_distances_nm = self._pair_distances_nm[np.isfinite(self._pair_distances_nm)]
        centers = np.asarray(result.get("subunit_centers", np.empty((0, 3))))
        self._subunit_template_ids = np.zeros(centers.shape[0], dtype=np.int64)
        for structure in result.get("structures", []):
            indices = np.asarray(structure.get("unit_indices", []), dtype=int)
            valid = indices[(indices >= 0) & (indices < centers.shape[0])]
            self._subunit_template_ids[valid] = int(structure.get("id", 0))
        self._raw_color_attributes = _raw_color_attributes_for_result(
            result, source_dataset, prefs)
        self._scatter_color_by = (
            "tid" if "tid" in self._raw_color_attributes
            else next(iter(self._raw_color_attributes), "")
        )
        self._scatter_colormap = "jet"
        self._scatter_black_background = False
        self._scatter_pages: dict[str, dict] = {}
        self._pair_geometry_cache = None
        reference = centers
        if not reference.shape[0]:
            reference = np.asarray(result.get("points_nm", np.empty((0, 3))), dtype=float)
        finite_reference = reference[np.all(np.isfinite(reference), axis=1)]
        self._scatter_view_center_nm = (
            np.mean(finite_reference, axis=0) if finite_reference.shape[0]
            else np.zeros(3, dtype=float)
        )
        self._scatter_view_scale_nm_per_px: float | None = None
        self._current_scatter_view: str | None = None
        # 2-D data (E.coli border-removal path) → use the flat XY scatter.
        self._prefer_2d = bool(result.get("is_2d", False)) if prefer_2d is None else bool(prefer_2d)
        self.setWindowTitle(f"HlyB Subunit Pair Analysis — {title}" if title else
                            "HlyB Subunit Pair Analysis")
        self.setWindowFlags(Qt.WindowType.Window)
        self.resize(1100, 900)

        root = QVBoxLayout(self)
        root.addWidget(self._summary_label())

        splitter = QSplitter(Qt.Orientation.Vertical)
        scatter_panel = self._build_scatter()
        histogram_panel = self._build_histogram()
        scatter_panel.setMinimumHeight(330)
        histogram_panel.setMinimumHeight(230)
        splitter.addWidget(scatter_panel)
        splitter.addWidget(histogram_panel)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)
        self._result_splitter = splitter
        QTimer.singleShot(0, lambda: splitter.setSizes([540, 300]))

    # -- summary --------------------------------------------------------

    def _summary_label(self) -> QLabel:
        r = self._result
        pd = self._pair_distances_nm
        text = ""
        if r.get("is_2d") and "n_total_traces" in r:
            text += (f"Traces: {r['n_total_traces']} total, {r['n_border_traces']} border-excluded, "
                     f"{r['n_traces']} interior   |   ")
            stats = r.get("cell_mask_stats") or {}
            if stats:
                shrink = (
                    f"{stats.get('border_fraction', 0):.2f} of half-width"
                    if str(stats.get("border_mode")) == "relative"
                    else f"{stats.get('border_size_nm', 0):.0f} nm"
                )
                text += (
                    f"Cells: {stats.get('n_cells', 0)} "
                    f"({stats.get('in_mask_fraction', 0):.0%} of locs delineated, "
                    f"median half-width {stats.get('median_half_width_nm', 0):.0f} nm); "
                    f"shrink {shrink} keeps {stats.get('retained_fraction', 0):.0%}   |   "
                )
        else:
            text += f"Traces: {r['n_traces']}   |   "
        if r.get("template_matching"):
            qc = r.get("match_qc", {})
            text += (
                f"Sub-units: {r['n_subunits']}   |   "
                f"Template matches: {r['n_structures']}   |   Pairs: {pd.size}   |   "
                f"unit Ø {r['dunit_nm']:.1f} nm, tolerance {r['model_pair_tolerance_nm']:.1f} nm   |   "
                f"tested {qc.get('n_candidates_tested', 0)}, "
                f"passed {qc.get('n_candidates_passed_thresholds', 0)}, "
                f"overlap-rejected {qc.get('n_overlap_rejected', 0)}"
            )
        else:
            text += (
                f"Sub-units: {r['n_subunits']}   |   "
                f"HlyB structures: {r['n_structures']}   |   Pairs: {pd.size}   |   "
                f"unit Ø {r['dunit_nm']:.1f} nm, HlyB radius {r['hlyb_diameter_nm']:.1f} nm"
            )
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl.setStyleSheet("font-weight: bold;")
        return lbl

    # -- scatter --------------------------------------------------------

    _AXIS_NAMES = {0: "x", 1: "y", 2: "z"}
    _VIEW_AXES = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}

    def _build_scatter(self) -> QWidget:
        """Scatter area with shared object controls for all four projections."""
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        row.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["XY", "XZ", "YZ", "3D"])
        self._view_combo.setCurrentText("XY" if self._prefer_2d else "3D")
        row.addWidget(self._view_combo)
        self._raw_loc_checkbox = QCheckBox("raw loc")
        self._subunit_detection_checkbox = QCheckBox("sub-unit detection")
        self._template_match_checkbox = QCheckBox("template match")
        self._pair_link_checkbox = QCheckBox("pair link")
        for checkbox in (
            self._raw_loc_checkbox,
            self._subunit_detection_checkbox,
            self._template_match_checkbox,
            self._pair_link_checkbox,
        ):
            checkbox.setChecked(True)
            checkbox.toggled.connect(self._refresh_scatter_display)
            row.addWidget(checkbox)
        self._raw_loc_checkbox.setToolTip("Show or hide the raw localization points.")
        self._subunit_detection_checkbox.setToolTip(
            "Show or hide detected subunit centres.")
        self._template_match_checkbox.setToolTip(
            "Color accepted subunits by template membership; uncheck to render "
            "every detected subunit in gray.")
        self._pair_link_checkbox.setToolTip(
            "Show or hide accepted pair links and their distance labels together.")
        row.addStretch(1)
        outer.addLayout(row)

        self._view_stack = QStackedWidget()
        self._view_pages: dict[str, int] = {}
        outer.addWidget(self._view_stack, 1)

        self._view_combo.currentTextChanged.connect(self._show_view)
        self._show_view(self._view_combo.currentText())
        return container

    def _show_view(self, view: str) -> None:
        previous_view = self._current_scatter_view
        if previous_view is not None and previous_view != view:
            self._capture_scatter_view_state(previous_view)
        if view not in self._view_pages:
            if view == "3D":
                try:
                    widget = self._build_gl_scatter(view_name=view)
                except Exception as exc:  # noqa: BLE001 - fall back to a flat XY scatter
                    widget = self._build_2d_scatter(
                        view_name=view, axes=(0, 1), reason=str(exc))
            else:
                widget = self._build_2d_scatter(
                    view_name=view, axes=self._VIEW_AXES[view])
            self._view_pages[view] = self._view_stack.addWidget(widget)
        self._view_stack.setCurrentIndex(self._view_pages[view])
        self._current_scatter_view = view
        if previous_view is None:
            QTimer.singleShot(
                0, lambda value=view: self._capture_scatter_view_state(value)
                if self._current_scatter_view == value else None,
            )
        elif previous_view != view:
            self._apply_scatter_view_state(view)
            QTimer.singleShot(
                0, lambda value=view: self._apply_scatter_view_state(value)
                if self._current_scatter_view == value else None,
            )
        page = self._scatter_pages.get(view)
        if page and page.get("kind") == "2d":
            QTimer.singleShot(0, lambda value=view: self._refresh_2d_pair_labels(value))
        elif page and page.get("kind") == "3d":
            QTimer.singleShot(0, lambda value=view: self._refresh_gl_pair_labels(value))

    @staticmethod
    def _vector_components(vector) -> np.ndarray:
        return np.asarray([vector.x(), vector.y(), vector.z()], dtype=float)

    @staticmethod
    def _page_pixel_size(page: dict) -> tuple[float, float]:
        if page.get("kind") == "3d":
            widget = page["view"]
            return max(float(widget.width()), 1.0), max(float(widget.height()), 1.0)
        view_box = page["plot"].getPlotItem().vb
        width = float(view_box.width())
        height = float(view_box.height())
        if width <= 1.0 or height <= 1.0:
            viewport = page["plot"].viewport()
            width, height = float(viewport.width()), float(viewport.height())
        return max(width, 1.0), max(height, 1.0)

    def _capture_scatter_view_state(self, view_name: str) -> None:
        """Capture the current projection as a shared world centre and pixel scale."""
        page = self._scatter_pages.get(view_name)
        if not page:
            return
        width, height = self._page_pixel_size(page)
        if page.get("kind") == "3d":
            opts = page["view"].opts
            center = self._vector_components(opts["center"]) + page["anchor"]
            distance = float(opts.get("distance", 0.0))
            fov = float(opts.get("fov", 60.0))
            vertical_span = 2.0 * distance * np.tan(np.deg2rad(fov) * 0.5)
            scale = vertical_span / height
        else:
            (x0, x1), (y0, y1) = page["plot"].getPlotItem().vb.viewRange()
            axes = page["axes"]
            center = self._scatter_view_center_nm.copy()
            center[axes[0]] = 0.5 * (float(x0) + float(x1))
            center[axes[1]] = 0.5 * (float(y0) + float(y1))
            scale = max(abs(float(x1) - float(x0)) / width,
                        abs(float(y1) - float(y0)) / height)
        if np.all(np.isfinite(center)):
            self._scatter_view_center_nm = center
        if np.isfinite(scale) and scale > 0.0:
            self._scatter_view_scale_nm_per_px = float(scale)

    def _apply_scatter_view_state(self, view_name: str) -> None:
        """Apply the shared centre/zoom to another 2-D projection or 3-D camera."""
        page = self._scatter_pages.get(view_name)
        scale = self._scatter_view_scale_nm_per_px
        if not page or scale is None or not np.isfinite(scale) or scale <= 0.0:
            return
        width, height = self._page_pixel_size(page)
        center = self._scatter_view_center_nm
        if page.get("kind") == "3d":
            view = page["view"]
            fov = float(view.opts.get("fov", 60.0))
            denominator = 2.0 * np.tan(np.deg2rad(fov) * 0.5)
            distance = max(float(scale) * height / max(denominator, 1e-12), 1e-6)
            local_center = center - page["anchor"]
            view.setCameraPosition(
                pos=pg.Vector(*[float(value) for value in local_center]),
                distance=distance,
            )
            self._refresh_gl_pair_labels(view_name)
            return
        axes = page["axes"]
        x_center, y_center = float(center[axes[0]]), float(center[axes[1]])
        x_half = 0.5 * float(scale) * width
        y_half = 0.5 * float(scale) * height
        page["plot"].getPlotItem().vb.setRange(
            xRange=(x_center - x_half, x_center + x_half),
            yRange=(y_center - y_half, y_center + y_half),
            padding=0.0,
        )
        self._refresh_2d_pair_labels(view_name)

    def _pair_segments(self) -> tuple[np.ndarray, list[tuple]]:
        """Flat segment endpoints plus endpoint/midpoint/distance records."""
        if self._pair_geometry_cache is not None:
            return self._pair_geometry_cache
        seg = []
        labels = []
        for st in self._result["structures"]:
            uc = np.asarray(st["unit_centers"], dtype=float)
            pair_idx = st.get("pair_index")
            if pair_idx is None:
                pair_idx = np.array([
                    (i, j) for i in range(uc.shape[0] - 1)
                    for j in range(i + 1, uc.shape[0])
                ])
            pair_dist = st.get("pair_distances")
            for p, (i, j) in enumerate(np.asarray(pair_idx, dtype=int)):
                p1, p2 = uc[i], uc[j]
                seg.append(p1)
                seg.append(p2)
                dist = (
                    float(pair_dist[p])
                    if pair_dist is not None and p < len(pair_dist)
                    else float(np.linalg.norm(p2 - p1))
                )
                labels.append((p1, p2, (p1 + p2) / 2.0, dist, int(st.get("id", 0))))
        self._pair_geometry_cache = (
            np.array(seg, dtype=np.float64) if seg else np.empty((0, 3)),
            labels,
        )
        return self._pair_geometry_cache

    def _scatter_mapped_rgba(self, values: np.ndarray, *, alpha: float = 1.0) -> np.ndarray:
        from ..colormaps import make_colormap

        vals = np.asarray(values, dtype=float).ravel()
        if not vals.size:
            return np.empty((0, 4), dtype=np.float32)
        finite = vals[np.isfinite(vals)]
        if not finite.size:
            norm = np.zeros(vals.size, dtype=float)
        else:
            lo, hi = (float(value) for value in np.nanpercentile(finite, [1.0, 99.0]))
            norm = np.zeros(vals.size, dtype=float) if hi <= lo else np.clip(
                (vals - lo) / (hi - lo), 0.0, 1.0)
            norm = np.nan_to_num(norm)
        cmap = make_colormap(self._scatter_colormap)
        rgba = np.asarray(cmap.map(norm, mode=pg.ColorMap.FLOAT), dtype=np.float32)
        if rgba.ndim != 2 or rgba.shape[1] < 4:
            rgba = np.column_stack([rgba[:, :3], np.ones(rgba.shape[0])])
        rgba[:, 3] = float(alpha)
        return rgba

    def _raw_point_colors(self, source_indices: np.ndarray) -> np.ndarray:
        source_indices = np.asarray(source_indices, dtype=np.int64).ravel()
        values = self._raw_color_attributes.get(self._scatter_color_by)
        if (values is not None and source_indices.size
                and np.all(source_indices >= 0)
                and np.all(source_indices < np.asarray(values).size)):
            return self._scatter_mapped_rgba(
                np.asarray(values, dtype=float)[source_indices], alpha=0.72)
        gray = (0.72, 0.72, 0.72, 0.58) if self._scatter_black_background else (
            0.20, 0.20, 0.20, 0.58)
        return np.tile(np.asarray(gray, dtype=np.float32), (source_indices.size, 1))

    def _subunit_colors(self) -> np.ndarray:
        centers = np.asarray(self._result["subunit_centers"], dtype=float)
        gray = (0.68, 0.68, 0.68, 0.95) if self._scatter_black_background else (
            0.48, 0.48, 0.48, 0.95)
        colors = np.tile(np.asarray(gray, dtype=np.float32), (centers.shape[0], 1))
        if not self._template_match_checkbox.isChecked() or not centers.shape[0]:
            return colors
        template_ids = sorted(
            int(value) for value in np.unique(self._subunit_template_ids) if value > 0)
        for ordinal, template_id in enumerate(template_ids):
            color = pg.intColor(ordinal, hues=max(len(template_ids), 9))
            colors[self._subunit_template_ids == template_id] = (
                color.redF(), color.greenF(), color.blueF(), 1.0)
        return colors

    def _build_gl_scatter(self, *, view_name: str) -> QWidget:
        import pyqtgraph.opengl as gl

        points = np.asarray(self._result["points_nm"], dtype=np.float64)
        centers = np.asarray(self._result["subunit_centers"], dtype=np.float64)
        seg, _labels = self._pair_segments()

        anchor = centers.mean(axis=0) if centers.shape[0] else (
            points.mean(axis=0) if points.shape[0] else np.zeros(3))

        owner = self

        class _InteractiveResultGLView(gl.GLViewWidget):
            def wheelEvent(self, event) -> None:  # noqa: N802
                super().wheelEvent(event)
                QTimer.singleShot(0, lambda: owner._refresh_gl_pair_labels(view_name))

            def mouseMoveEvent(self, event) -> None:  # noqa: N802
                super().mouseMoveEvent(event)
                owner._on_gl_subunit_hover(view_name, event)

            def mouseReleaseEvent(self, event) -> None:  # noqa: N802
                super().mouseReleaseEvent(event)
                QTimer.singleShot(0, lambda: owner._refresh_gl_pair_labels(view_name))

            def resizeEvent(self, event) -> None:  # noqa: N802
                super().resizeEvent(event)
                QTimer.singleShot(0, lambda: owner._refresh_gl_pair_labels(view_name))

        view = _InteractiveResultGLView()
        view.setMouseTracking(True)
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(
            lambda pos, widget=view: self._show_scatter_context_menu(pos, widget))
        view.setBackgroundColor("k" if self._scatter_black_background else "w")

        raw = points
        raw_indices = np.arange(points.shape[0], dtype=np.int64)
        if raw.shape[0] > _MAX_RAW_POINTS:
            step = int(np.ceil(raw.shape[0] / _MAX_RAW_POINTS))
            raw = raw[::step]
            raw_indices = raw_indices[::step]
        raw_item = gl.GLScatterPlotItem(pxMode=True, size=2.5)
        subunit_item = gl.GLScatterPlotItem(pxMode=True, size=10.0)
        link_item = gl.GLLinePlotItem(
            mode="lines", width=1.5, antialias=False)
        view.addItem(raw_item)
        view.addItem(link_item)
        view.addItem(subunit_item)

        ref = centers if centers.shape[0] else points
        grid = gl.GLGridItem()
        view.addItem(grid)
        if ref.shape[0]:
            span = float(np.linalg.norm(np.ptp(ref - anchor, axis=0))) or 100.0
            grid.setSize(span * 1.5, span * 1.5)
            grid.setSpacing(max(span / 10.0, 1.0), max(span / 10.0, 1.0))
            view.setCameraPosition(distance=max(span * 1.6, 100.0))
        self._scatter_pages[view_name] = {
            "kind": "3d",
            "widget": view,
            "view": view,
            "gl": gl,
            "anchor": anchor,
            "raw_points": raw,
            "raw_indices": raw_indices,
            "raw_item": raw_item,
            "centers": centers,
            "subunit_item": subunit_item,
            "segments": seg,
            "link_item": link_item,
            "grid": grid,
            "label_items": {},
            "span": float(np.linalg.norm(np.ptp(ref - anchor, axis=0))) if ref.shape[0] else 100.0,
        }
        self._refresh_3d_scatter_page(view_name)
        QTimer.singleShot(0, lambda: self._refresh_gl_pair_labels(view_name))
        return view

    def _xy_origin_top_left(self) -> bool:
        """Whether Y increases downward, as in the render and scatter views."""
        value = str(
            (self._prefs.get("plot", {}) or {}).get("scatter_xy_origin", "top_left")
        ).lower()
        return value != "bottom_left"

    def _apply_y_axis_direction(self, view_name: str, plot) -> None:
        """Match the Loc Scatter Plot and Render view Y-axis convention.

        Those default to a top-left origin (Y increasing downward, the
        ImageJ/MATLAB image convention), and only in the XY orientation: XZ and
        YZ keep a natural Y so the axial coordinate reads upward. Without this
        the same cell appeared mirrored here relative to every other view of it.
        """
        try:
            invert = view_name == "XY" and self._xy_origin_top_left()
            plot.getPlotItem().getViewBox().invertY(invert)
        except Exception:
            pass

    def _build_2d_scatter(
        self,
        *,
        view_name: str,
        axes: tuple[int, int] = (0, 1),
        reason: str = "",
    ) -> QWidget:
        ax0, ax1 = axes
        name0, name1 = self._AXIS_NAMES[ax0], self._AXIS_NAMES[ax1]
        plot = pg.PlotWidget(
            background="k" if self._scatter_black_background else "w")
        plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        plot.customContextMenuRequested.connect(
            lambda pos, widget=plot: self._show_scatter_context_menu(pos, widget))
        plot.getPlotItem().setMenuEnabled(False)
        try:
            plot.getPlotItem().vb.setMenuEnabled(False)
        except Exception:
            pass
        plot.setAspectLocked(True)
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel("bottom", name0, units="nm")
        plot.setLabel("left", name1, units="nm")
        self._apply_y_axis_direction(view_name, plot)
        if reason:
            plot.setTitle(f"3-D view unavailable ({reason}); showing {name0.upper()}{name1.upper()}")
        elif self._result.get("is_2d"):
            plot.setTitle("Interior (gray) vs border-excluded (orange) localizations; "
                          "sub-unit centres and measured pairs")

        border_item = pg.ScatterPlotItem(size=2, pen=None)
        raw_item = pg.ScatterPlotItem(size=2.5, pen=None)
        link_item = pg.PlotDataItem()
        subunit_item = pg.ScatterPlotItem(size=10, hoverable=True, tip=None)
        plot.addItem(border_item)
        plot.addItem(raw_item)
        plot.addItem(link_item)
        plot.addItem(subunit_item)
        subunit_item.sigHovered.connect(self._on_subunit_hover)

        border = self._result.get("border_points_nm")
        border = np.asarray(border, dtype=float) if border is not None else np.empty((0, 3))
        points = np.asarray(self._result["points_nm"], dtype=np.float64)
        raw_indices = np.arange(points.shape[0], dtype=np.int64)
        if points.shape[0] > _MAX_RAW_POINTS:
            step = int(np.ceil(points.shape[0] / _MAX_RAW_POINTS))
            points = points[::step]
            raw_indices = raw_indices[::step]
        centers = np.asarray(self._result["subunit_centers"], dtype=np.float64)
        seg, _labels = self._pair_segments()
        line_x, line_y = self._projected_segment_lines(seg, axes)
        self._scatter_pages[view_name] = {
            "kind": "2d",
            "widget": plot,
            "plot": plot,
            "axes": axes,
            "border_points": border,
            "border_item": border_item,
            "raw_points": points,
            "raw_indices": raw_indices,
            "raw_item": raw_item,
            "raw_brush_key": None,
            "raw_brushes": None,
            "centers": centers,
            "subunit_item": subunit_item,
            "link_item": link_item,
            "line_x": line_x,
            "line_y": line_y,
            "label_items": {},
        }
        plot.getPlotItem().vb.sigRangeChanged.connect(
            lambda *_args, value=view_name: self._refresh_2d_pair_labels(value))
        self._refresh_2d_scatter_page(view_name)
        plot.autoRange()
        self._refresh_2d_pair_labels(view_name)
        return plot

    @staticmethod
    def _projected_segment_lines(
        segments: np.ndarray, axes: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not segments.shape[0]:
            return np.empty(0), np.empty(0)
        pairs = np.asarray(segments, dtype=float).reshape(-1, 2, 3)
        nan = np.full((pairs.shape[0], 1), np.nan)
        x = np.column_stack([pairs[:, 0, axes[0]], pairs[:, 1, axes[0]], nan]).ravel()
        y = np.column_stack([pairs[:, 0, axes[1]], pairs[:, 1, axes[1]], nan]).ravel()
        return x, y

    def _show_scatter_context_menu(self, pos, source: QWidget) -> None:
        """Loc-Scatter-style menu for the HlyB result scatter pane."""
        from ..colormaps import named_colormap_names
        from ..colors import solid_color_names

        menu = QMenu(self)
        view_menu = menu.addMenu("View as")
        for view in ("XY", "XZ", "YZ", "3D"):
            action = view_menu.addAction(view)
            action.setCheckable(True)
            action.setChecked(view == self._view_combo.currentText())
            action.triggered.connect(
                lambda _checked=False, value=view: self._view_combo.setCurrentText(value))

        color_menu = menu.addMenu("Color by")
        if self._raw_color_attributes:
            for value in self._raw_color_attributes:
                action = color_menu.addAction(value)
                action.setCheckable(True)
                action.setChecked(value == self._scatter_color_by)
                action.triggered.connect(
                    lambda _checked=False, selected=value: self._set_scatter_color_by(selected))
        else:
            unavailable = color_menu.addAction("No aligned attributes available")
            unavailable.setEnabled(False)

        cmap_menu = menu.addMenu("Colormap")
        for value in named_colormap_names():
            action = cmap_menu.addAction(value)
            action.setCheckable(True)
            action.setChecked(value == self._scatter_colormap)
            action.triggered.connect(
                lambda _checked=False, selected=value: self._set_scatter_colormap(selected))
        cmap_menu.addSeparator()
        solid_menu = cmap_menu.addMenu("Solid color")
        for color_name in solid_color_names():
            value = f"solid:{color_name}"
            action = solid_menu.addAction(color_name)
            action.setCheckable(True)
            action.setChecked(value == self._scatter_colormap)
            action.triggered.connect(
                lambda _checked=False, selected=value: self._set_scatter_colormap(selected))
        solid_menu.addSeparator()
        custom = solid_menu.addAction("Custom...")
        custom.setCheckable(True)
        custom.setChecked(self._scatter_colormap.startswith("solid:custom:"))
        custom.triggered.connect(self._pick_scatter_solid_color)

        background = menu.addAction("Black Background")
        background.setCheckable(True)
        background.setChecked(self._scatter_black_background)
        background.triggered.connect(self._set_scatter_black_background)

        menu.addSeparator()
        menu.addAction("Reset View", self._reset_scatter_view)
        menu.exec(source.mapToGlobal(pos))

    def _set_scatter_color_by(self, value: str) -> None:
        if value not in self._raw_color_attributes:
            return
        self._scatter_color_by = value
        self._refresh_scatter_display()

    def _set_scatter_colormap(self, value: str) -> None:
        self._scatter_colormap = str(value)
        self._refresh_scatter_display()

    def _pick_scatter_solid_color(self) -> None:
        from PyQt6.QtWidgets import QColorDialog

        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self._set_scatter_colormap(f"solid:custom:{color.name()}")

    def _set_scatter_black_background(self, enabled: bool) -> None:
        self._scatter_black_background = bool(enabled)
        self._refresh_scatter_display()

    def _refresh_scatter_display(self, *_args) -> None:
        for view_name, page in tuple(self._scatter_pages.items()):
            if page.get("kind") == "3d":
                self._refresh_3d_scatter_page(view_name)
            else:
                self._refresh_2d_scatter_page(view_name)

    @staticmethod
    def _rgba_brushes(rgba: np.ndarray) -> list:
        rgba = np.asarray(rgba, dtype=float)
        return [
            pg.mkBrush(
                int(np.clip(color[0], 0, 1) * 255),
                int(np.clip(color[1], 0, 1) * 255),
                int(np.clip(color[2], 0, 1) * 255),
                int(np.clip(color[3], 0, 1) * 255),
            )
            for color in rgba
        ]

    @staticmethod
    def _rgba_brush(color: np.ndarray):
        color = np.asarray(color, dtype=float)
        return pg.mkBrush(
            int(np.clip(color[0], 0, 1) * 255),
            int(np.clip(color[1], 0, 1) * 255),
            int(np.clip(color[2], 0, 1) * 255),
            int(np.clip(color[3], 0, 1) * 255),
        )

    def _refresh_2d_scatter_page(self, view_name: str) -> None:
        page = self._scatter_pages.get(view_name)
        if not page or page.get("kind") != "2d":
            return
        plot = page["plot"]
        black = self._scatter_black_background
        plot.setBackground("k" if black else "w")
        axes = page["axes"]

        border = page["border_points"]
        if border.ndim == 2 and border.shape[0]:
            page["border_item"].setData(
                x=border[:, axes[0]], y=border[:, axes[1]],
                brush=pg.mkBrush(230, 160, 40, 105), pen=None, size=2.5,
            )
        page["border_item"].setVisible(self._raw_loc_checkbox.isChecked())

        raw = page["raw_points"]
        if raw.shape[0]:
            raw_colors = self._raw_point_colors(page["raw_indices"])
            if self._scatter_color_by not in self._raw_color_attributes:
                raw_brushes = self._rgba_brush(raw_colors[0])
            else:
                brush_key = (
                    self._scatter_color_by, self._scatter_colormap,
                    self._scatter_black_background, raw.shape[0],
                )
                if page["raw_brush_key"] != brush_key:
                    page["raw_brush_key"] = brush_key
                    page["raw_brushes"] = self._rgba_brushes(raw_colors)
                raw_brushes = page["raw_brushes"]
            page["raw_item"].setData(
                x=raw[:, axes[0]], y=raw[:, axes[1]],
                brush=raw_brushes,
                pen=None, size=2.5,
            )
        page["raw_item"].setVisible(self._raw_loc_checkbox.isChecked())

        centers = page["centers"]
        if centers.shape[0]:
            outline = pg.mkPen("w" if black else "k", width=0.7)
            page["subunit_item"].setData(
                x=centers[:, axes[0]], y=centers[:, axes[1]],
                data=np.arange(centers.shape[0], dtype=int),
                brush=self._rgba_brushes(self._subunit_colors()),
                pen=outline, size=10, hoverable=True, tip=None,
            )
        page["subunit_item"].setVisible(
            self._subunit_detection_checkbox.isChecked())

        link_color = (80, 165, 255, 230) if black else (35, 95, 205, 220)
        page["link_item"].setData(
            page["line_x"], page["line_y"], connect="finite",
            pen=pg.mkPen(link_color, width=1.35),
        )
        page["link_item"].setVisible(self._pair_link_checkbox.isChecked())
        self._refresh_2d_pair_labels(view_name)

    def _refresh_2d_pair_labels(self, view_name: str) -> None:
        page = self._scatter_pages.get(view_name)
        if not page or page.get("kind") != "2d":
            return
        existing = page["label_items"]
        if not self._pair_link_checkbox.isChecked():
            for item in existing.values():
                item.setVisible(False)
            return

        (x0, x1), (y0, y1) = page["plot"].getPlotItem().vb.viewRange()
        width = max(int(page["plot"].viewport().width()), 1)
        height = max(int(page["plot"].viewport().height()), 1)
        if x1 <= x0 or y1 <= y0:
            return
        ax0, ax1 = page["axes"]
        _seg, labels = self._pair_segments()
        candidates = []
        for index, (p1, p2, mid, _distance, _template_id) in enumerate(labels):
            mx, my = float(mid[ax0]), float(mid[ax1])
            if not (x0 <= mx <= x1 and y0 <= my <= y1):
                continue
            dx = abs(float(p2[ax0] - p1[ax0])) / (x1 - x0) * width
            dy = abs(float(p2[ax1] - p1[ax1])) / (y1 - y0) * height
            pixel_length = float(np.hypot(dx, dy))
            if pixel_length >= _PAIR_LABEL_MIN_PIXELS:
                candidates.append((pixel_length, index))
        selected = {
            index for _length, index in sorted(candidates, reverse=True)[:_MAX_DISTANCE_LABELS]
        }
        color = (225, 225, 225) if self._scatter_black_background else (30, 30, 30)
        fill = (pg.mkBrush(0, 0, 0, 145) if self._scatter_black_background
                else pg.mkBrush(255, 255, 255, 155))
        for index, item in existing.items():
            item.setVisible(index in selected)
            if index in selected:
                item.setColor(color)
                item.fill = fill
                item.update()
        for index in selected - existing.keys():
            _p1, _p2, mid, distance, _template_id = labels[index]
            item = pg.TextItem(
                f"{distance:.1f}", color=color, anchor=(0.5, 0.5),
                fill=fill,
            )
            item.setZValue(20)
            item.setPos(float(mid[ax0]), float(mid[ax1]))
            page["plot"].addItem(item, ignoreBounds=True)
            existing[index] = item

    def _subunit_tooltip(self, index: int) -> str:
        centers = np.asarray(self._result["subunit_centers"], dtype=float)
        if not (0 <= index < centers.shape[0]):
            return ""
        x, y, z = centers[index, :3]
        template_id = int(self._subunit_template_ids[index])
        template_text = str(template_id) if template_id > 0 else "—"
        return (
            f"X: {x:.2f} (nm)\n"
            f"Y: {y:.2f} (nm)\n"
            f"Z: {z:.2f} (nm)\n"
            f"Template ID: {template_text}"
        )

    def _on_subunit_hover(self, _item, points, event) -> None:
        if (not self._subunit_detection_checkbox.isChecked()
                or points is None or len(points) == 0):
            QToolTip.hideText()
            return
        try:
            index = int(points[0].data())
            screen = event.screenPos()
            pos = QPoint(round(screen.x()), round(screen.y()))
        except Exception:
            from PyQt6.QtGui import QCursor
            index = int(points[0].data())
            pos = QCursor.pos()
        tooltip = self._subunit_tooltip(index)
        if tooltip:
            QToolTip.showText(pos, tooltip, self)

    def _refresh_3d_scatter_page(self, view_name: str) -> None:
        page = self._scatter_pages.get(view_name)
        if not page or page.get("kind") != "3d":
            return
        black = self._scatter_black_background
        page["view"].setBackgroundColor("k" if black else "w")
        try:
            page["grid"].setColor(pg.mkColor(175, 175, 175, 130) if black
                                  else pg.mkColor(70, 70, 70, 105))
        except Exception:
            pass
        anchor = page["anchor"]
        raw = page["raw_points"]
        page["raw_item"].setData(
            pos=(raw - anchor).astype(np.float32),
            color=self._raw_point_colors(page["raw_indices"]), size=2.5, pxMode=True,
        )
        page["raw_item"].setVisible(self._raw_loc_checkbox.isChecked())
        centers = page["centers"]
        page["subunit_item"].setData(
            pos=(centers - anchor).astype(np.float32),
            color=self._subunit_colors(), size=10.0, pxMode=True,
        )
        page["subunit_item"].setVisible(
            self._subunit_detection_checkbox.isChecked())
        segments = page["segments"]
        link_color = (0.35, 0.68, 1.0, 0.95) if black else (0.12, 0.36, 0.82, 0.9)
        page["link_item"].setData(
            pos=(segments - anchor).astype(np.float32), color=link_color,
            width=1.5, mode="lines", antialias=False,
        )
        page["link_item"].setVisible(self._pair_link_checkbox.isChecked())
        for item in (page["raw_item"], page["subunit_item"], page["link_item"]):
            try:
                item.setGLOptions("translucent")
            except Exception:
                pass
        self._refresh_gl_pair_labels(view_name)

    @staticmethod
    def _project_gl_points(page: dict, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from PyQt6.QtGui import QVector4D

        view = page["view"]
        width, height = max(view.width(), 1), max(view.height(), 1)
        viewport = (0, 0, width, height)
        matrix = view.projectionMatrix(viewport, viewport) * view.viewMatrix()
        screen = np.full((len(points), 2), np.nan, dtype=float)
        visible = np.zeros(len(points), dtype=bool)
        for index, point in enumerate(np.asarray(points, dtype=float)):
            clip = matrix * QVector4D(float(point[0]), float(point[1]), float(point[2]), 1.0)
            w = float(clip.w())
            if not np.isfinite(w) or w <= 0:
                continue
            ndc = np.array([clip.x() / w, clip.y() / w, clip.z() / w], dtype=float)
            if not np.all(np.isfinite(ndc)):
                continue
            screen[index] = ((ndc[0] + 1.0) * width * 0.5,
                             (1.0 - ndc[1]) * height * 0.5)
            visible[index] = (
                -1.0 <= ndc[0] <= 1.0
                and -1.0 <= ndc[1] <= 1.0
                and -1.0 <= ndc[2] <= 1.0
            )
        return screen, visible

    def _refresh_gl_pair_labels(self, view_name: str) -> None:
        page = self._scatter_pages.get(view_name)
        if not page or page.get("kind") != "3d":
            return
        existing = page["label_items"]
        if not self._pair_link_checkbox.isChecked():
            for item in existing.values():
                item.setVisible(False)
            return
        _seg, labels = self._pair_segments()
        if not labels:
            return
        anchor = page["anchor"]
        endpoints = np.asarray([
            point - anchor
            for p1, p2, _mid, _distance, _template_id in labels
            for point in (p1, p2)
        ], dtype=float)
        projected, visible = self._project_gl_points(page, endpoints)
        projected = projected.reshape(-1, 2, 2)
        visible = visible.reshape(-1, 2)
        lengths = np.linalg.norm(projected[:, 1] - projected[:, 0], axis=1)
        candidates = np.where(
            visible.all(axis=1) & (lengths >= _PAIR_LABEL_MIN_PIXELS))[0]
        if candidates.size > _MAX_DISTANCE_LABELS:
            order = np.argsort(lengths[candidates])[::-1][:_MAX_DISTANCE_LABELS]
            candidates = candidates[order]
        selected = set(int(index) for index in candidates)
        color = (230, 230, 230, 255) if self._scatter_black_background else (25, 25, 25, 255)
        for index, item in existing.items():
            item.setVisible(index in selected)
            if index in selected:
                item.setData(color=color)
        gl = page["gl"]
        for index in selected - existing.keys():
            _p1, _p2, mid, distance, _template_id = labels[index]
            item = gl.GLTextItem(
                pos=np.asarray(mid - anchor, dtype=float),
                text=f"{distance:.1f}", color=color,
            )
            item.setDepthValue(20)
            page["view"].addItem(item)
            existing[index] = item

    def _on_gl_subunit_hover(self, view_name: str, event) -> None:
        page = self._scatter_pages.get(view_name)
        if (not page or page.get("kind") != "3d"
                or not self._subunit_detection_checkbox.isChecked()
                or event.buttons() != Qt.MouseButton.NoButton):
            QToolTip.hideText()
            return
        centers = page["centers"] - page["anchor"]
        if not centers.shape[0]:
            QToolTip.hideText()
            return
        projected, visible = self._project_gl_points(page, centers)
        cursor = event.position()
        distances = np.linalg.norm(
            projected - np.array([cursor.x(), cursor.y()], dtype=float), axis=1)
        distances[~visible] = np.inf
        index = int(np.argmin(distances))
        if not np.isfinite(distances[index]) or distances[index] > 11.0:
            QToolTip.hideText()
            return
        global_pos = page["view"].mapToGlobal(event.position().toPoint())
        QToolTip.showText(global_pos, self._subunit_tooltip(index), self)

    def _reset_scatter_view(self) -> None:
        view_name = self._view_combo.currentText()
        page = self._scatter_pages.get(view_name)
        if not page:
            return
        if page.get("kind") == "3d":
            self._reset_gl_camera(view_name)
        else:
            page["plot"].autoRange()
            self._refresh_2d_pair_labels(view_name)
            QTimer.singleShot(
                0, lambda value=view_name: self._capture_scatter_view_state(value))

    def _reset_gl_camera(self, view_name: str) -> None:
        page = self._scatter_pages.get(view_name)
        if not page or page.get("kind") != "3d":
            return
        page["view"].opts["center"] = pg.Vector(0.0, 0.0, 0.0)
        page["view"].setCameraPosition(
            distance=max(float(page.get("span", 100.0)) * 1.6, 100.0),
            elevation=30.0, azimuth=45.0,
        )
        self._capture_scatter_view_state(view_name)
        self._refresh_gl_pair_labels(view_name)

    # -- histogram ------------------------------------------------------

    def _build_histogram(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Bin size:"))
        self._distance_bin_spin = QDoubleSpinBox()
        self._distance_bin_spin.setRange(0.1, 10.0)
        self._distance_bin_spin.setSingleStep(0.1)
        # Keep enough precision to display the existing automatic bin width;
        # user changes still happen in 0.1 nm increments.
        self._distance_bin_spin.setDecimals(2)
        self._distance_bin_spin.setSuffix(" nm")
        self._distance_bin_spin.setToolTip("Distance histogram bin size (0.1–10 nm).")
        controls.addWidget(self._distance_bin_spin)
        self._show_all_pairs_checkbox = QCheckBox("show all (remove template gating)")
        self._show_all_pairs_checkbox.setToolTip(
            "Overlay all pair distances among detected subunit centers before HlyB template assignment."
        )
        self._show_all_pairs_checkbox.toggled.connect(self._on_show_all_pairs_toggled)
        controls.addWidget(self._show_all_pairs_checkbox)
        self._show_lognormal_fit_checkbox = QCheckBox("show fit")
        self._show_lognormal_fit_checkbox.setToolTip(
            "Fit the all-pair histogram when it is shown, otherwise the "
            "template-gated one."
        )
        self._show_lognormal_fit_checkbox.toggled.connect(
            self._on_show_lognormal_fit_toggled)
        controls.addWidget(self._show_lognormal_fit_checkbox)
        self._fit_shape_combo = QComboBox()
        self._fit_shape_combo.addItems(list(self.FIT_SHAPES))
        self._fit_shape_combo.setToolTip(
            "Functional form fitted to the distance histogram.\n\n"
            "3-D blurred distance — the exact distribution of a fixed separation\n"
            "seen through isotropic localization error. This is what a rigid\n"
            "dimer actually produces, and it is the only one of the three that\n"
            "recovers the true distance when that distance is comparable to the\n"
            "blur.\n\n"
            "Gaussian — its large-distance limit. Close for well-separated\n"
            "sites, biased high when the blur is not small.\n\n"
            "lognormal — kept for comparison with earlier results. A rigid\n"
            "separation in 3-D is orientation-independent, so nothing about the\n"
            "geometry makes it log-normal; on simulated dimers it fits 2-5x\n"
            "worse than a Gaussian and reports a distance that is too large.")
        self._fit_shape_combo.currentTextChanged.connect(
            lambda _t: self._render_distance_histogram(
                bin_size=float(self._distance_bin_spin.value())))
        controls.addWidget(self._fit_shape_combo)
        controls.addStretch(1)
        self._distance_stats_label = QLabel()
        self._distance_stats_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        controls.addWidget(self._distance_stats_label)
        outer.addLayout(controls)

        plot = pg.PlotWidget(background="w")
        self._distance_hist_plot = plot
        plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        plot.customContextMenuRequested.connect(self._show_distance_plot_context_menu)
        plot_item = plot.getPlotItem()
        plot_item.setMenuEnabled(False)
        self._distance_view_box = plot_item.vb
        try:
            self._distance_view_box.setMenuEnabled(False)
        except Exception:
            pass
        self._distance_original_mouse_drag_event = self._distance_view_box.mouseDragEvent
        self._distance_view_box.mouseDragEvent = self._distance_zoom_mouse_drag_event
        plot_item.autoBtnClicked = lambda *args, **kwargs: self._reset_distance_view()
        outer.addWidget(plot, 1)

        pd = self._pair_distances_nm
        centers = np.asarray(
            self._result.get("subunit_centers", np.empty((0, 3))), dtype=float)
        can_show_all = (
            centers.ndim == 2
            and int(np.all(np.isfinite(centers), axis=1).sum()) >= 2
        )
        self._show_all_pairs_checkbox.setEnabled(can_show_all)
        if pd.size:
            _, auto_edges = self._distance_histogram_data(pd)
            auto_width = float(np.diff(auto_edges).mean()) if auto_edges.size > 1 else 1.0
            self._distance_bin_spin.setValue(float(np.clip(auto_width, 0.1, 10.0)))
            self._render_distance_histogram(bin_size=None)
        else:
            self._distance_bin_spin.setValue(1.0 if can_show_all else 0.1)
            self._distance_bin_spin.setEnabled(can_show_all)
            self._show_lognormal_fit_checkbox.setEnabled(False)
            self._fit_shape_combo.setEnabled(False)
            self._render_distance_histogram(bin_size=None)
        self._distance_reset_bin_size = float(self._distance_bin_spin.value())
        self._distance_bin_spin.valueChanged.connect(self._on_distance_bin_size_changed)
        return container

    @staticmethod
    def _distance_histogram_data(values: np.ndarray, bin_size: float | None = None):
        """Return histogram counts and edges for the requested distance width."""
        values = np.asarray(values, dtype=float).ravel()
        values = values[np.isfinite(values) & (values >= 0)]
        if not values.size:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        if bin_size is None:
            auto_edges = np.histogram_bin_edges(values, bins="auto")
            width = float(np.diff(auto_edges).mean()) if auto_edges.size > 1 else 1.0
            width = float(np.clip(width, 0.1, 10.0))
        else:
            width = float(bin_size)

        if not np.isfinite(width) or width <= 0:
            width = 1.0
        upper = _adaptive_histogram_upper_nm(values, width)
        n_bins = max(1, int(np.ceil(upper / width)))
        edges = width * np.arange(n_bins + 1, dtype=float)
        return np.histogram(values, bins=edges)

    def _on_distance_bin_size_changed(self, value: float) -> None:
        if not self._distance_zoom_rebinning:
            self._distance_reset_bin_size = float(value)
        previous_range = (
            self._distance_view_box.viewRange()
            if self._distance_view_box is not None else None
        )
        self._render_distance_histogram(bin_size=float(value))
        if previous_range is not None:
            self._distance_view_box.setRange(
                xRange=previous_range[0], yRange=previous_range[1], padding=0.0)

    def _on_show_all_pairs_toggled(self, checked: bool) -> None:
        can_fit = bool(checked or self._pair_distances_nm.size)
        self._show_lognormal_fit_checkbox.setEnabled(can_fit)
        self._fit_shape_combo.setEnabled(can_fit)
        if not can_fit:
            self._show_lognormal_fit_checkbox.setChecked(False)
        self._render_distance_histogram(bin_size=float(self._distance_bin_spin.value()))

    def _on_show_lognormal_fit_toggled(self, checked: bool) -> None:
        self._lognormal_fit_requested = bool(checked)
        previous_range = (
            self._distance_view_box.viewRange()
            if self._distance_view_box is not None else None
        )
        self._render_distance_histogram(bin_size=float(self._distance_bin_spin.value()))
        if previous_range is not None:
            self._distance_view_box.setRange(
                xRange=previous_range[0], yRange=previous_range[1], padding=0.0)

    def _render_distance_histogram(self, *, bin_size: float | None) -> None:
        plot = self._distance_hist_plot
        if plot is None:
            return
        plot.clear()
        self._distance_bar_item = None
        self._all_distance_bar_item = None
        self._lognormal_fit_result = None
        self._lognormal_curve_item = None
        self._lognormal_mean_line = None
        self._lognormal_fit_text = None
        plot.showGrid(x=True, y=True, alpha=0.2)
        plot.setLabel("bottom", "sub-unit pair distance", units="nm")
        plot.setLabel("left", "count")
        plot.setTitle("Distance among sub-unit pairs (within each HlyB structure)")

        pd = self._pair_distances_nm
        counts, edges = self._distance_histogram_data(pd, bin_size)
        width = (
            float(np.diff(edges).mean())
            if edges.size > 1
            else float(bin_size or self._distance_bin_spin.value() or 1.0)
        )
        all_counts = None
        if self._show_all_pairs_checkbox.isChecked():
            all_hist = self._all_pair_histogram(width)
            if all_hist is not None and np.asarray(all_hist["counts"]).size:
                all_counts = np.asarray(all_hist["counts"], dtype=np.int64)
                edges = np.asarray(all_hist["edges_nm"], dtype=float)
                counts = np.histogram(pd, bins=edges)[0]

        if all_counts is None and (not pd.size or not counts.size):
            plot.setTitle("No sub-unit pairs found")
            if self._show_all_pairs_checkbox.isChecked() and self._all_pair_histogram_error:
                self._distance_stats_label.setText(
                    f"All-to-all calculation failed: {self._all_pair_histogram_error}"
                )
            else:
                self._distance_stats_label.setText("No pair distances")
            return

        if all_counts is not None:
            plot.setTitle("Template-gated and all detected subunit pair distances")

        centers = 0.5 * (edges[:-1] + edges[1:])
        if all_counts is not None:
            self._all_distance_bar_item = _HoverHistogramBarItem(
                counts=all_counts, edges=edges, series_name="All detected subunits",
                x=centers, height=all_counts, width=width * 0.96,
                brush=pg.mkBrush(175, 175, 175, 125),
                pen=pg.mkPen(145, 145, 145, 180),
            )
            plot.addItem(self._all_distance_bar_item)
        self._distance_bar_item = _HoverHistogramBarItem(
            counts=counts, edges=edges, series_name="Template-gated matches",
            x=centers, height=counts, width=width * (0.68 if all_counts is not None else 0.95),
            brush=pg.mkBrush(80, 130, 200, 180), pen=pg.mkPen(60, 100, 160))
        plot.addItem(self._distance_bar_item)

        if pd.size:
            peak_idx = int(np.argmax(counts))
            peak_value = float(0.5 * (edges[peak_idx] + edges[peak_idx + 1]))
            peak_count = int(counts[peak_idx])
            stats = f"Max-count bin: {peak_value:.2f} nm ({peak_count:,})"
        else:
            stats = "Template-gated: no pairs"
        if all_counts is not None:
            all_peak_idx = int(np.argmax(all_counts))
            all_peak_value = float(0.5 * (edges[all_peak_idx] + edges[all_peak_idx + 1]))
            all_peak_count = int(all_counts[all_peak_idx])
            stats += (
                f"   |   All max-count bin: {all_peak_value:.2f} nm "
                f"({all_peak_count:,})"
            )
        elif self._show_all_pairs_checkbox.isChecked() and self._all_pair_histogram_error:
            stats += f"   |   All-to-all calculation failed: {self._all_pair_histogram_error}"
        self._distance_stats_label.setText(stats)

        # Use 0–40 nm for short HlyB-only results, but expand when the analysis
        # genuinely returns larger distances instead of drawing empty bins.
        plot.setXRange(0.0, float(edges[-1]), padding=0.0)
        if self._lognormal_fit_requested:
            fit_counts = all_counts if all_counts is not None else counts
            fit_series = "all pairs" if all_counts is not None else "template-gated"
            self._draw_lognormal_fit(fit_counts, edges, series_name=fit_series)
        self._fit_distance_histogram_view()

    # -- distance-histogram right-click zoom --------------------------

    DISTANCE_ZOOM_MODES = ("horizontal", "vertical", "unconstrained")

    def _show_distance_plot_context_menu(self, pos) -> None:
        """Replace pyqtgraph's menu with the Attribute Histogram zoom tools."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        zoom_menu = menu.addMenu("Zoom")
        zoom_menu.setToolTipsVisible(True)
        zoom_menu.menuAction().setToolTip(
            "Arm a zoom drag. The tool releases itself after one drag,\n"
            "so the next left-drag pans as usual. Esc cancels."
        )
        for mode in self.DISTANCE_ZOOM_MODES:
            action = zoom_menu.addAction(mode)
            action.setCheckable(True)
            action.setChecked(self._distance_zoom_mode == mode)
            action.setToolTip(_DISTANCE_ZOOM_TOOLTIPS[mode])
            action.triggered.connect(
                lambda _checked=False, value=mode: self._toggle_distance_zoom_mode(value)
            )
        menu.addSeparator()
        reset_action = menu.addAction("Reset View")
        reset_action.setToolTip(
            "Show the complete distance distribution at the current bin size."
        )
        reset_action.triggered.connect(self._reset_distance_view)
        menu.exec(self._distance_hist_plot.mapToGlobal(pos))

    def _toggle_distance_zoom_mode(self, mode: str) -> None:
        if mode not in self.DISTANCE_ZOOM_MODES:
            return
        self._set_distance_zoom_mode(
            None if self._distance_zoom_mode == mode else mode)

    def _set_distance_zoom_mode(self, mode: str | None) -> None:
        self._distance_zoom_mode = mode
        self._clear_distance_zoom_preview()
        try:
            viewport = self._distance_hist_plot.viewport()
            if mode:
                viewport.setCursor(Qt.CursorShape.CrossCursor)
            else:
                viewport.unsetCursor()
        except Exception:
            pass

    def _reset_distance_view(self) -> None:
        self._set_distance_zoom_mode(None)
        target = float(
            self._distance_reset_bin_size
            if self._distance_reset_bin_size is not None
            else self._distance_bin_spin.value()
        )
        self._distance_bin_spin.blockSignals(True)
        try:
            self._distance_bin_spin.setValue(target)
        finally:
            self._distance_bin_spin.blockSignals(False)
        self._render_distance_histogram(
            bin_size=float(self._distance_bin_spin.value()))

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() == Qt.Key.Key_Escape and self._distance_zoom_mode:
            self._set_distance_zoom_mode(None)
            event.accept()
            return
        super().keyPressEvent(event)

    def _distance_zoom_mouse_drag_event(self, event, axis=None) -> None:
        if (not self._distance_zoom_mode
                or event.button() != Qt.MouseButton.LeftButton):
            self._distance_original_mouse_drag_event(event, axis=axis)
            return
        event.accept()
        if event.isStart():
            self._distance_zoom_drag_start = self._distance_view_box.mapSceneToView(
                event.buttonDownScenePos(Qt.MouseButton.LeftButton)
            )
            self._clear_distance_zoom_preview()
            self._distance_zoom_preview = pg.PlotDataItem(
                pen=pg.mkPen(
                    (30, 120, 220), width=1.5, style=Qt.PenStyle.DashLine)
            )
            self._distance_zoom_preview.setZValue(30)
            self._distance_hist_plot.addItem(
                self._distance_zoom_preview, ignoreBounds=True)
        if self._distance_zoom_drag_start is None:
            return
        current = self._distance_view_box.mapSceneToView(event.scenePos())
        self._update_distance_zoom_preview(
            self._distance_zoom_drag_start, current)
        if event.isFinish():
            self._apply_distance_zoom_drag(
                self._distance_zoom_drag_start, current)
            self._distance_zoom_drag_start = None
            self._set_distance_zoom_mode(None)

    def _update_distance_zoom_preview(self, start, current) -> None:
        if self._distance_zoom_preview is None:
            return
        x0, x1 = float(start.x()), float(current.x())
        y0, y1 = float(start.y()), float(current.y())
        (vx0, vx1), (vy0, vy1) = self._distance_view_box.viewRange()
        if self._distance_zoom_mode == "horizontal":
            cap = (vy1 - vy0) * 0.08
            self._distance_zoom_preview.setData(
                [x0, x1, np.nan, x0, x0, np.nan, x1, x1],
                [y1, y1, np.nan, y1 - cap, y1 + cap, np.nan, y1 - cap, y1 + cap],
            )
        elif self._distance_zoom_mode == "vertical":
            cap = (vx1 - vx0) * 0.08
            self._distance_zoom_preview.setData(
                [x1, x1, np.nan, x1 - cap, x1 + cap, np.nan, x1 - cap, x1 + cap],
                [y0, y1, np.nan, y0, y0, np.nan, y1, y1],
            )
        else:
            self._distance_zoom_preview.setData(
                [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])

    def _apply_distance_zoom_drag(self, start, current) -> None:
        x0, x1 = sorted((float(start.x()), float(current.x())))
        y0, y1 = sorted((float(start.y()), float(current.y())))
        (vx0, vx1), (vy0, vy1) = self._distance_view_box.viewRange()
        min_dx = abs(vx1 - vx0) * 1e-6
        min_dy = abs(vy1 - vy0) * 1e-6
        if self._distance_zoom_mode == "horizontal":
            if (x1 - x0) > min_dx:
                self._distance_zoom_to(x0, x1, None, None, rebin=True)
        elif self._distance_zoom_mode == "vertical":
            y0 = max(y0, 0.0)
            if (y1 - y0) > min_dy:
                self._distance_zoom_to(vx0, vx1, y0, y1, rebin=False)
        elif (x1 - x0) > min_dx and (y1 - y0) > min_dy:
            self._distance_zoom_to(
                x0, x1, max(y0, 0.0), y1, rebin=True, auto_y=True)

    def _distance_zoom_to(
        self,
        x0: float,
        x1: float,
        y0: float | None,
        y1: float | None,
        *,
        rebin: bool,
        auto_y: bool = False,
    ) -> None:
        if rebin:
            width = self._distance_zoom_bin_width(x0, x1)
            if width is not None:
                before = float(self._distance_bin_spin.value())
                self._distance_zoom_rebinning = True
                try:
                    self._distance_bin_spin.setValue(width)
                finally:
                    self._distance_zoom_rebinning = False
                if np.isclose(before, float(self._distance_bin_spin.value())):
                    self._render_distance_histogram(
                        bin_size=float(self._distance_bin_spin.value()))
        if y0 is None or y1 is None or auto_y:
            fitted = self._distance_auto_y_for_x_range(x0, x1)
            if fitted is not None:
                y0, y1 = fitted
        if y0 is None or y1 is None:
            _x, (y0, y1) = self._distance_view_box.viewRange()
        y0 = max(float(y0), 0.0)
        y1 = float(y1)
        if y1 <= y0:
            y1 = y0 + max(abs(y0) * 0.01, 1.0)
        self._distance_view_box.setRange(
            xRange=(x0, x1), yRange=(y0, y1), padding=0.0)

    def _distance_auto_y_for_x_range(
        self, x0: float, x1: float,
    ) -> tuple[float, float] | None:
        peaks = []
        for item in (self._all_distance_bar_item, self._distance_bar_item):
            if item is None:
                continue
            counts = np.asarray(item._counts, dtype=float)
            edges = np.asarray(item._edges, dtype=float)
            if edges.size != counts.size + 1:
                continue
            visible = (edges[1:] >= x0) & (edges[:-1] <= x1)
            if visible.any():
                peaks.append(float(np.max(counts[visible])))
        if self._lognormal_curve_item is not None:
            curve_x, curve_y = self._lognormal_curve_item.getData()
            curve_x = np.asarray(curve_x, dtype=float)
            curve_y = np.asarray(curve_y, dtype=float)
            visible = (curve_x >= x0) & (curve_x <= x1) & np.isfinite(curve_y)
            if visible.any():
                peaks.append(float(np.max(curve_y[visible])))
        if not peaks or max(peaks) <= 0:
            return None
        return 0.0, max(peaks) * (1.0 + _DISTANCE_ZOOM_Y_HEADROOM)

    def _distance_zoom_bin_width(self, x0: float, x1: float) -> float | None:
        span = float(x1) - float(x0)
        if not np.isfinite([x0, x1]).all() or span <= 0:
            return None
        item = (
            self._all_distance_bar_item
            if self._show_all_pairs_checkbox.isChecked()
            and self._all_distance_bar_item is not None
            else self._distance_bar_item
        )
        if item is None:
            return None
        counts = np.asarray(item._counts, dtype=np.int64)
        edges = np.asarray(item._edges, dtype=float)
        if edges.size != counts.size + 1:
            return None
        visible = (edges[1:] >= x0) & (edges[:-1] <= x1)
        n_values = int(np.sum(counts[visible])) if visible.any() else 0
        target = int(np.clip(
            n_values // _DISTANCE_ZOOM_VALUES_PER_BIN,
            _DISTANCE_ZOOM_MIN_BINS,
            _DISTANCE_ZOOM_MAX_BINS,
        ))
        width = span / target
        width = float(np.clip(
            width,
            self._distance_bin_spin.minimum(),
            self._distance_bin_spin.maximum(),
        ))
        return width if np.isfinite(width) and width > 0 else None

    def _fit_distance_histogram_view(self) -> None:
        xmax = 0.0
        for item in (self._all_distance_bar_item, self._distance_bar_item):
            if item is not None and item._edges.size:
                xmax = max(xmax, float(item._edges[-1]))
        if self._lognormal_fit_result is not None:
            xmax = max(xmax, 1.1 * float(self._lognormal_fit_result["mean_nm"]))
        if xmax <= 0:
            return
        fitted = self._distance_auto_y_for_x_range(0.0, xmax)
        y_range = fitted if fitted is not None else (0.0, 1.0)
        self._distance_view_box.setRange(
            xRange=(0.0, xmax), yRange=y_range, padding=0.0)

    def _clear_distance_zoom_preview(self) -> None:
        if self._distance_zoom_preview is not None:
            try:
                self._distance_hist_plot.removeItem(
                    self._distance_zoom_preview)
            except Exception:
                pass
            self._distance_zoom_preview = None

    def _all_pair_histogram(self, width: float) -> dict | None:
        key = round(float(width), 6)
        cached = self._all_pair_histogram_cache.get(key)
        if cached is not None:
            return cached

        centers = np.asarray(
            self._result.get("subunit_centers", np.empty((0, 3))), dtype=float)
        finite_centers = centers[np.all(np.isfinite(centers), axis=1)]
        if finite_centers.shape[0] < 2:
            return None

        n_pairs = finite_centers.shape[0] * (finite_centers.shape[0] - 1) // 2
        self._distance_stats_label.setText(
            f"Computing {n_pairs:,} all-to-all subunit pairs…"
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            from ..analysis.hlyb_clustering import all_pair_distance_histogram

            self._all_pair_histogram_error = None
            hist = all_pair_distance_histogram(finite_centers, width)
            if not hist["counts"].size:
                return None
            upper = _adaptive_histogram_upper_nm(
                np.asarray([hist["max_distance_nm"]], dtype=float), width)
            n_bins = max(1, int(np.ceil(upper / width)))
            counts = np.asarray(hist["counts"], dtype=np.int64)
            if counts.size < n_bins:
                counts = np.pad(counts, (0, n_bins - counts.size))
            elif counts.size > n_bins:
                n_bins = counts.size
            hist = dict(hist)
            hist["counts"] = counts
            hist["edges_nm"] = width * np.arange(n_bins + 1, dtype=float)
            self._all_pair_histogram_cache[key] = hist
            return hist
        except Exception as exc:
            self._all_pair_histogram_error = str(exc)
            return None
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

    #: Functional forms offered for the distance histogram, best first.
    #:
    #: A rigid separation measured in three dimensions is NOT log-normal.  The
    #: separation is orientation-independent, so placing dimers on a curved
    #: surface does not skew it at all; the only broadening is the localization
    #: error, which is additive and symmetric.  The exact result is the
    #: distribution of the magnitude of a 3-D Gaussian-perturbed vector — a
    #: non-central chi with three degrees of freedom — which tends to a Gaussian
    #: once the distance exceeds a few times the blur.
    #:
    #: Measured on simulated fixed-distance dimers on a capsule surface, the
    #: log-normal is 2-5x worse in RMSE than a Gaussian at every distance and
    #: blur tried, and it biases the reported mean upward (11.3 nm for a true
    #: 10.0 nm at 10 nm localization sigma).  A plain Gaussian is much better
    #: but still biased when the distance is comparable to the blur (14.2 nm for
    #: a true 10.0 nm at 25 nm sigma), because a symmetric form cannot represent
    #: a positive quantity bounded below by zero.  The exact form recovers
    #: 9.9 nm there.  It is therefore the default; the others are kept so a fit
    #: made earlier can be reproduced and compared.
    FIT_SHAPES = ("3-D blurred distance", "Gaussian", "lognormal")

    @staticmethod
    def _blurred_distance_histogram_fit(counts: np.ndarray, edges: np.ndarray) -> dict:
        """Fit the exact 3-D blurred-distance density to bin counts."""
        from scipy.optimize import curve_fit

        from ..analysis.hlyb_pairwise import offset_gaussian_pdf

        counts = np.asarray(counts, dtype=float).ravel()
        edges = np.asarray(edges, dtype=float).ravel()
        if edges.size != counts.size + 1:
            raise ValueError("Histogram edges do not align with counts.")
        centers = 0.5 * (edges[:-1] + edges[1:])
        usable = np.isfinite(centers) & np.isfinite(counts) & (centers > 0)
        if int(np.count_nonzero(usable & (counts > 0))) < 3:
            raise ValueError("At least three occupied positive-distance bins are required.")
        weights = np.maximum(counts[usable], 1e-9)
        d0 = float(np.average(centers[usable], weights=weights))
        s0 = max(float(np.sqrt(np.average((centers[usable] - d0) ** 2, weights=weights))), 0.5)

        def model(x, amplitude, distance, sigma):
            return amplitude * offset_gaussian_pdf(np.asarray(x, dtype=float),
                                                   float(distance), float(sigma))

        params, _ = curve_fit(
            model, centers[usable], counts[usable],
            p0=(max(float(counts.sum()) * float(np.diff(edges).mean()), 1.0), d0, s0),
            bounds=((0.0, 0.05, 0.05), (np.inf, 1e4, 1e3)), maxfev=20_000)
        amplitude, distance, sigma = (float(v) for v in params)
        predicted = model(centers[usable], amplitude, distance, sigma)
        resid = float(np.sqrt(np.mean((counts[usable] - predicted) ** 2)))
        return {
            "shape": "3-D blurred distance",
            "model": lambda x, a, d, s: model(x, a, d, s),
            "amplitude": amplitude, "mu": distance, "sigma": sigma,
            "mean_nm": distance, "rmse_counts": resid,
            "report": (f"distance = {distance:.2f} nm\nblur sigma = {sigma:.2f} nm"),
        }

    @staticmethod
    def _gaussian_histogram_fit(counts: np.ndarray, edges: np.ndarray) -> dict:
        """Least-squares fit of ``A*N(x; mean, sd)`` to bin counts."""
        from scipy.optimize import curve_fit

        counts = np.asarray(counts, dtype=float).ravel()
        edges = np.asarray(edges, dtype=float).ravel()
        if edges.size != counts.size + 1:
            raise ValueError("Histogram edges do not align with counts.")
        centers = 0.5 * (edges[:-1] + edges[1:])
        usable = np.isfinite(centers) & np.isfinite(counts)
        if int(np.count_nonzero(usable & (counts > 0))) < 3:
            raise ValueError("At least three occupied bins are required.")
        weights = np.maximum(counts[usable], 1e-9)
        m0 = float(np.average(centers[usable], weights=weights))
        s0 = max(float(np.sqrt(np.average((centers[usable] - m0) ** 2, weights=weights))), 0.5)

        def model(x, amplitude, mean, sd):
            x = np.asarray(x, dtype=float)
            return amplitude * np.exp(-0.5 * ((x - mean) / sd) ** 2) / (
                sd * np.sqrt(2.0 * np.pi))

        params, _ = curve_fit(
            model, centers[usable], counts[usable],
            p0=(max(float(counts.sum()) * float(np.diff(edges).mean()), 1.0), m0, s0),
            bounds=((0.0, -1e4, 0.05), (np.inf, 1e4, 1e3)), maxfev=20_000)
        amplitude, mean, sd = (float(v) for v in params)
        predicted = model(centers[usable], amplitude, mean, sd)
        resid = float(np.sqrt(np.mean((counts[usable] - predicted) ** 2)))
        return {
            "shape": "Gaussian",
            "model": model, "amplitude": amplitude, "mu": mean, "sigma": sd,
            "mean_nm": mean, "rmse_counts": resid,
            "report": f"mean = {mean:.2f} nm\nsd = {sd:.2f} nm",
        }

    def _fit_shape(self) -> str:
        if self._fit_shape_combo is None:
            return self.FIT_SHAPES[0]
        return str(self._fit_shape_combo.currentText()) or self.FIT_SHAPES[0]

    def _distance_histogram_fit(self, counts, edges, shape: str) -> dict:
        if shape == "Gaussian":
            return self._gaussian_histogram_fit(counts, edges)
        if shape == "lognormal":
            return self._lognormal_histogram_fit(counts, edges)
        return self._blurred_distance_histogram_fit(counts, edges)

    @staticmethod
    def _lognormal_histogram_fit(counts: np.ndarray, edges: np.ndarray) -> dict:
        """Least-squares fit of ``A*lognormal(x; mu, sigma)`` to bin counts."""
        from scipy.optimize import curve_fit

        counts = np.asarray(counts, dtype=float).ravel()
        edges = np.asarray(edges, dtype=float).ravel()
        if edges.size != counts.size + 1:
            raise ValueError("Histogram edges do not align with counts.")
        centers = 0.5 * (edges[:-1] + edges[1:])
        usable = np.isfinite(centers) & np.isfinite(counts) & (centers > 0)
        occupied = usable & (counts > 0)
        if int(np.count_nonzero(occupied)) < 3:
            raise ValueError("At least three occupied positive-distance bins are required.")

        log_x = np.log(centers[occupied])
        weights = counts[occupied]
        mu0 = float(np.average(log_x, weights=weights))
        sigma0 = float(np.sqrt(np.average((log_x - mu0) ** 2, weights=weights)))
        sigma0 = max(sigma0, 0.1)
        width = float(np.diff(edges).mean())
        amplitude0 = max(float(counts.sum()) * width, 1.0)

        def model(x, amplitude, mu, sigma):
            x = np.asarray(x, dtype=float)
            out = np.zeros_like(x)
            positive = x > 0
            xp = x[positive]
            out[positive] = (
                amplitude
                * np.exp(-0.5 * ((np.log(xp) - mu) / sigma) ** 2)
                / (xp * sigma * np.sqrt(2.0 * np.pi))
            )
            return out

        x_fit = centers[usable]
        y_fit = counts[usable]
        params, _ = curve_fit(
            model,
            x_fit,
            y_fit,
            p0=(amplitude0, mu0, sigma0),
            bounds=((0.0, -20.0, 0.01), (np.inf, 20.0, 5.0)),
            maxfev=20_000,
        )
        amplitude, mu, sigma = (float(v) for v in params)
        predicted = model(x_fit, amplitude, mu, sigma)
        rmse = float(np.sqrt(np.mean((predicted - y_fit) ** 2)))
        mean_nm = float(np.exp(mu + 0.5 * sigma * sigma))
        return {
            "shape": "lognormal",
            "amplitude": amplitude,
            "mu": mu,
            "sigma": sigma,
            "mean_nm": mean_nm,
            "rmse_counts": rmse,
            "model": model,
            "report": f"mean = {mean_nm:.2f} nm\nμ = {mu:.4f}\nσ = {sigma:.4f}",
        }

    def _draw_lognormal_fit(
        self,
        counts: np.ndarray,
        edges: np.ndarray,
        *,
        series_name: str,
    ) -> None:
        counts = np.asarray(counts, dtype=float).ravel()
        edges = np.asarray(edges, dtype=float).ravel()
        width = float(np.diff(edges).mean()) if edges.size > 1 else 0.0
        shape = self._fit_shape()
        cache_key = (
            str(series_name), shape, round(width, 6), int(counts.size),
            int(np.sum(counts)), float(edges[-1]) if edges.size else 0.0,
        )
        fit = self._lognormal_fit_cache.get(cache_key)
        if fit is None:
            try:
                fit = self._distance_histogram_fit(counts, edges, shape)
            except Exception as exc:
                self._distance_stats_label.setText(
                    f"{self._distance_stats_label.text()}   |   "
                    f"{shape} fit unavailable: {exc}"
                )
                return
            self._lognormal_fit_cache[cache_key] = fit

        self._lognormal_fit_result = fit
        x_curve = np.linspace(max(float(edges[0]), 1e-6), float(edges[-1]), 1200)
        y_curve = fit["model"](
            x_curve, fit["amplitude"], fit["mu"], fit["sigma"])
        fit_color = (215, 90, 25)
        self._lognormal_curve_item = self._distance_hist_plot.plot(
            x_curve, y_curve, pen=pg.mkPen(fit_color, width=2.0))
        self._lognormal_mean_line = pg.InfiniteLine(
            pos=fit["mean_nm"], angle=90,
            pen=pg.mkPen(fit_color, width=1.5, style=Qt.PenStyle.DashLine),
        )
        self._distance_hist_plot.addItem(self._lognormal_mean_line)

        y_top = max(float(np.max(counts)), float(np.max(y_curve)), 1.0)
        anchor_x = 1.0 if fit["mean_nm"] > 0.5 * float(edges[-1]) else 0.0
        text = (
            f"{fit.get('shape', 'fit')} fit ({series_name})\n"
            f"{fit.get('report', '')}\n"
            f"fit RMSE = {fit['rmse_counts']:.3g} counts/bin"
        )
        self._lognormal_fit_text = pg.TextItem(
            text=text, color=(90, 45, 20), anchor=(anchor_x, 0.0),
            border=pg.mkPen(fit_color), fill=pg.mkBrush(255, 248, 238, 225),
        )
        self._lognormal_fit_text.setPos(fit["mean_nm"], 0.94 * y_top)
        self._distance_hist_plot.addItem(self._lognormal_fit_text)
        if fit["mean_nm"] > float(edges[-1]):
            self._distance_hist_plot.setXRange(
                0.0, 1.1 * fit["mean_nm"], padding=0.0)
        self._distance_stats_label.setText(
            f"{self._distance_stats_label.text()}   |   "
            f"{fit.get('shape', 'Fit')}: {fit['mean_nm']:.2f} nm"
        )
