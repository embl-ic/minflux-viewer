"""
Shared plumbing for the ``mfv`` API namespaces.

Every namespace is a thin object bound to the live application state. It holds
the :class:`AppState` and a resolver for the main window, and nothing else --
namespaces are stateless views onto the application, so two facades (a script
and a plugin) can coexist without sharing anything but the app itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..core.app_state import AppState
    from ..core.dataset import MinfluxDataset


class ApiError(RuntimeError):
    """
    Raised for script/plugin-facing API misuse, with a message written for the
    person who wrote the script rather than for a maintainer.

    ``minflux_viewer.scripting.ScriptError`` is an alias of this, so existing
    scripts that catch ``mfv.ScriptError`` keep working.
    """


class Namespace:
    """Base for every ``mfv.<name>`` namespace."""

    #: Namespace attribute name on the facade (``mfv.data`` -> ``"data"``).
    name: str = ""

    def __init__(self, facade: Any) -> None:
        self._facade = facade

    # -- access to the application ------------------------------------------

    @property
    def _state(self) -> "AppState":
        return self._facade.state

    def _main_window(self):
        """
        The bound :class:`MainWindow`, or raise.

        Namespaces must call this rather than reaching for a stored window, so
        a facade created before the window exists (tests, headless use) fails
        with a clear message instead of an ``AttributeError``.
        """
        return self._facade.require_main_window()

    def _dataset(self, dataset=None) -> "MinfluxDataset":
        """Resolve *dataset* (None/index/name/object) to a dataset, or raise."""
        return self._facade.resolve_dataset(dataset)

    def _dataset_index(self, dataset=None) -> int:
        return self._facade.resolve_dataset_index(dataset)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<mfv.{self.name or type(self).__name__}>"


def _todo(track: str, what: str) -> "NotImplementedError":
    """
    Build the placeholder raised by an unimplemented stub.

    Phase 0 lands every signature so the parallel tracks compile and test
    against final names; each track then replaces the bodies it owns. The
    message names the owning track so a premature call is self-explaining.
    """
    return NotImplementedError(
        f"mfv: {what} is not implemented yet (owned by extension-layer track {track})."
    )
