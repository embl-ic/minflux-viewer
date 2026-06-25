"""
minflux_viewer.ui.layer_brightness
===================================
Reusable **per-layer brightness/contrast** controller built on the render
viewer's :class:`brightness_contrast_dialog.BrightnessContrastDialog`.

A viewer (e.g. the convolution segmentation tools) registers one or more named
image *layers* — each with a callback to fetch its pixel sample (for the
histogram), a callback to apply ``(lo, hi)`` levels to its ImageItem(s), an *auto*
level function and a *default* level function. The controller owns a single
always-on-top B/C palette and re-targets it at the chosen layer, so several layers
(Response, Data, …) share the same Fiji-style dialog.

Qt-free except for the lazily-imported dialog, so the level/auto/reset logic is
unit-testable without a running QApplication.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

LevelFn = Callable[[], "tuple[float, float] | None"]
PixelFn = Callable[[], "np.ndarray | None"]
ApplyFn = Callable[[float, float], None]


def percentile_levels(pixels, lo_pct: float = 0.0, hi_pct: float = 99.5,
                      *, positive_only: bool = False) -> "tuple[float, float] | None":
    """Auto contrast levels from a pixel sample (Fiji-style percentile stretch)."""
    vals = np.asarray(pixels, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if positive_only:
        vals = vals[vals > 0]
    if vals.size == 0:
        return None
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


class _Layer:
    __slots__ = ("get_pixels", "apply", "auto", "default", "levels")

    def __init__(self, get_pixels: PixelFn, apply: ApplyFn, auto: LevelFn,
                 default: LevelFn) -> None:
        self.get_pixels = get_pixels
        self.apply = apply
        self.auto = auto
        self.default = default
        self.levels: "tuple[float, float] | None" = None   # user-set levels


class LayerBrightness:
    """Drives one shared B/C dialog across several named image layers."""

    def __init__(self) -> None:
        self._dialog = None
        self._layers: dict[str, _Layer] = {}
        self._target: str | None = None

    # ------------------------------------------------------------- registry
    def add_layer(self, name: str, *, get_pixels: PixelFn, apply: ApplyFn,
                  auto: LevelFn, default: LevelFn) -> None:
        self._layers[name] = _Layer(get_pixels, apply, auto, default)
        if self._target is None:
            self._target = name

    def names(self) -> list[str]:
        return list(self._layers)

    def effective_levels(self, name: str) -> "tuple[float, float] | None":
        """Current levels for *name*: the user-set levels, else the default."""
        lay = self._layers.get(name)
        if lay is None:
            return None
        return lay.levels if lay.levels is not None else lay.default()

    # ------------------------------------------------------------- target
    def set_target(self, name: str) -> None:
        if name in self._layers:
            self._target = name
            if self.is_open():
                self._refresh()

    @property
    def target(self) -> str | None:
        return self._target

    # ------------------------------------------------------------- dialog
    def is_open(self) -> bool:
        return self._dialog is not None and self._dialog.isVisible()

    def open(self, parent=None) -> None:
        if self._dialog is None:
            from .brightness_contrast_dialog import BrightnessContrastDialog
            self._dialog = BrightnessContrastDialog(
                on_levels_changed=self._on_levels,
                on_auto=self._on_auto,
                on_reset=self._on_reset,
                parent=parent)
        self._refresh()
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()

    def close(self) -> None:
        if self._dialog is not None:
            try:
                self._dialog.close()
            except Exception:
                pass
            self._dialog = None

    def refresh_if_open(self) -> None:
        """Re-pull the current target's pixels/levels into the dialog (call after a
        recompute so the histogram reflects the new image)."""
        if self.is_open():
            self._refresh()

    # ------------------------------------------------------------- internals
    def _cur(self) -> "_Layer | None":
        return self._layers.get(self._target) if self._target else None

    def _refresh(self) -> None:
        lay = self._cur()
        if lay is None or self._dialog is None:
            return
        px = lay.get_pixels()
        self._dialog.set_data(px if px is not None else np.empty(0))
        lv = self.effective_levels(self._target)
        if lv is not None:
            self._dialog.set_levels(*lv)
        self._dialog.set_auto_state(False)
        self._dialog.setWindowTitle(f"Brightness / Contrast — {self._target}")

    def _on_levels(self, lo: float, hi: float) -> None:
        lay = self._cur()
        if lay is None:
            return
        lay.levels = (float(lo), float(hi))
        lay.apply(float(lo), float(hi))

    def _on_auto(self) -> None:
        lay = self._cur()
        if lay is None:
            return
        lv = lay.auto()
        if lv is None:
            return
        lay.levels = lv
        lay.apply(*lv)
        if self._dialog is not None:
            self._dialog.set_levels(*lv)
            self._dialog.set_auto_state(True)

    def _on_reset(self) -> None:
        lay = self._cur()
        if lay is None:
            return
        lv = lay.default()
        if lv is None:
            lv = lay.auto()
        lay.levels = None          # back to "follow default"
        if lv is not None:
            lay.apply(*lv)
            if self._dialog is not None:
                self._dialog.set_levels(*lv)
                self._dialog.set_auto_state(False)
