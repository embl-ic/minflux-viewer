"""
``mfv.view`` -- the dataset-owned viewer windows.

Every method here goes through a **public** ``MainWindow`` method. The private
``_show_*`` methods this namespace replaced are the reason the extension layer
exists: external code cannot be refactored alongside a leading underscore.

Viewer windows are dataset-owned -- one per dataset, reused -- so calling
:meth:`View.render` twice for the same dataset raises the existing window
rather than opening a second one.
"""

from __future__ import annotations

from typing import Any

from ._base import ApiError, Namespace

#: Render/scatter view orientations.
ORIENTATIONS = ("XY", "XZ", "YZ")


class View(Namespace):
    """Open and drive the standard viewer windows."""

    name = "view"

    def render(self, dataset: Any = None, *, orientation: str | None = None):
        """
        Open (or raise) the Render View for a dataset (default: active).

        ``orientation`` is one of :data:`ORIENTATIONS`; omitted leaves whatever
        the window already shows.
        """
        win = self._main_window().show_render(self._idx_or_none(dataset))
        if win is not None and orientation:
            self._set_orientation(win, orientation)
        return win

    def scatter(self, dataset: Any = None, *, color_by: str | None = None):
        """Open (or raise) the Localization Scatter Plot."""
        win = self._main_window().show_scatter(self._idx_or_none(dataset))
        if win is not None and color_by:
            self._set_combo(win, ("_cmap_combo", "_color_combo"), color_by, "colour-by")
        return win

    def histogram(self, dataset: Any = None, *, attribute: str | None = None):
        """Open (or raise) the Attribute Histogram."""
        win = self._main_window().show_histogram(self._idx_or_none(dataset))
        if win is not None and attribute:
            self._set_combo(win, ("_attr_combo",), attribute, "attribute")
        return win

    def attribute_plot(
        self,
        dataset: Any = None,
        *,
        x: str | None = None,
        y: str | None = None,
        z: str | None = None,
        c: str | None = None,
    ):
        """
        Open (or raise) the Attribute Plot.

        X and Y always exist; Z and C are independent optional dimensions, so
        XY, XYZ, XYC and XYZC are all reachable. ``None`` leaves a dimension as
        the window has it.
        """
        win = self._main_window().show_attribute_plot(self._idx_or_none(dataset))
        if win is None:
            return None
        for name, value, combo in (
            ("x", x, "_x_combo"), ("y", y, "_y_combo"),
            ("z", z, "_z_combo"), ("c", c, "_c_combo"),
        ):
            if value:
                self._set_combo(win, (combo,), value, f"{name} attribute")
        return win

    def console(self):
        """Open (or raise) the Console window."""
        return self._main_window().show_console()

    def script_editor(self):
        """Open (or raise) the Script Editor."""
        return self._main_window().show_script_editor()

    # -- appearance ----------------------------------------------------------

    def snapshot(self, path: str, *, dataset: Any = None, view: str = "render") -> str:
        """
        Write the current contents of a viewer window to a PNG.

        Captures what is on screen, at the window's own resolution -- it is not
        a re-render, so it reflects the exact zoom, filters and channels the
        user is looking at. Returns the path written.
        """
        opener = {"render": self.render, "scatter": self.scatter,
                  "histogram": self.histogram,
                  "attribute": self.attribute_plot}.get(str(view))
        if opener is None:
            raise ApiError(
                "view must be one of: render, scatter, histogram, attribute."
            )
        win = opener(dataset)
        if win is None:
            raise ApiError(f"No {view} window is available for that dataset.")
        pixmap = win.grab()
        if not pixmap.save(str(path)):
            raise ApiError(f"Could not write the snapshot to {path!r}.")
        return str(path)

    def volume(self, dataset: Any = None):
        """
        Open the 3-D volume preview.

        Deferred in API 1.0: the volume window is owned by its render window
        and is reached through that window's own View menu, so publishing it
        here would mean publishing a second ownership path.
        """
        raise NotImplementedError(
            "mfv.view.volume() is deferred in API 1.0 — open the 3-D preview "
            "from the render window's View menu."
        )

    def lut(self, dataset: Any = None, **kwargs) -> None:
        """
        Set the colour mapping of a dataset render channel.

        Deferred in API 1.0: colour state is split across per-dataset overlay
        LUTs, per-window levels and the shared LUT dialog, and publishing a
        single setter before that is unified would freeze the wrong shape.
        """
        raise NotImplementedError(
            "mfv.view.lut() is deferred in API 1.0 — use the LUT dialog."
        )

    def refresh(self, dataset: Any = None) -> None:
        """
        Tell open views that a dataset changed.

        The write helpers in ``mfv.data`` already notify; this is for code that
        changed something outside them.
        """
        idx = self._dataset_index(dataset)
        self._state.notify_attributes_changed(idx)
        self._state.notify_filter_changed(idx)

    # -- internal ------------------------------------------------------------

    def _idx_or_none(self, dataset: Any):
        """``None`` means 'the active dataset', which the openers understand."""
        return None if dataset is None else self._dataset_index(dataset)

    @staticmethod
    def _set_orientation(win, orientation: str) -> None:
        wanted = str(orientation).upper()
        if wanted not in ORIENTATIONS:
            raise ApiError(
                f"orientation must be one of {', '.join(ORIENTATIONS)}, not {orientation!r}."
            )
        setter = getattr(win, "_set_orientation", None)
        if callable(setter):
            setter(wanted)

    @staticmethod
    def _set_combo(win, candidates: tuple[str, ...], value: str, what: str) -> None:
        """
        Drive one of a window's selectors by its visible text.

        Selectors are still private widgets; until each window publishes a
        setter this reaches for the first name that exists and reports clearly
        when the value is not on offer, rather than failing silently.
        """
        for attr in candidates:
            combo = getattr(win, attr, None)
            if combo is None:
                continue
            if combo.findText(value) < 0:
                offered = [combo.itemText(i) for i in range(combo.count())]
                raise ApiError(
                    f"{value!r} is not an available {what}. "
                    f"Offered: {', '.join(offered) or '(none)'}"
                )
            combo.setCurrentText(value)
            return
        raise ApiError(f"This window has no {what} selector.")
