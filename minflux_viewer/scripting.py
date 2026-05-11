"""
minflux_viewer.scripting
========================
Small script-facing facade for ImageJ/Fiji-like automation.

The facade intentionally exposes stable application operations rather than
internal Qt window/controller details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .core.app_state import AppState


class MinfluxViewerFacade:
    """Stable, compact API intended for interactive scripts."""

    def __init__(self, state: "AppState") -> None:
        self._state = state

    def get_active_dataset(self):
        return self._state.active_dataset

    def get_datasets(self) -> list:
        return list(self._state.datasets)

    def get_active_scene(self) -> None:
        """Scene objects are planned but not implemented yet."""
        return None

    def plot_histogram(self, values: Any, *, bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Return histogram counts/edges for script prototypes."""
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return np.histogram(arr, bins=bins)

    def plot_scatter(self, x: Any, y: Any) -> dict[str, np.ndarray]:
        """Return a lightweight scatter handle placeholder for scripts."""
        return {
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
        }

    def render(self, component=None, *, selector: str = "last", px_nm: float = 5.0) -> dict[str, Any]:
        """
        Placeholder render facade.

        A real scene/render handle will replace this once the viewer layer API
        is formalized. Returning structured metadata keeps early scripts
        testable without exposing internal windows.
        """
        return {
            "component": component,
            "selector": selector,
            "px_nm": float(px_nm),
        }

    def show_image(self, image: Any) -> Any:
        """Placeholder hook for future image-window integration."""
        return image

    def new_scene(self) -> None:
        """Scene objects are planned but not implemented yet."""
        return None


def create_facade(state: "AppState") -> MinfluxViewerFacade:
    return MinfluxViewerFacade(state)
