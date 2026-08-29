"""External Python-library path integration seam.

Phase 0 installs this no-op before built-in or external plugins are imported.
Extension-layer Track D replaces the body with validated, append-only path
installation; keeping the public signature here lets that work stay isolated
from the main-window startup path.
"""

from __future__ import annotations


def install_paths(prefs: dict) -> list[str]:
    """Install configured external-library paths and return those accepted.

    The foundation implementation intentionally changes no import state.
    Track D supplies validation and the append-only ``sys.path`` behaviour.
    """
    del prefs
    return []
