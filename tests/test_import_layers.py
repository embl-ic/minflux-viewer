"""Which layer of the package may import which (Phase 4 slice 4.0).

``core`` imports none of the others; ``msr`` may import ``core``; ``analysis``
may import ``core`` and ``msr``; ``api`` reaches ``ui`` and ``plugins`` only
from inside a function, because it acts on a running application; ``ui``,
``plugins`` and ``__main__`` may import anything. ``utils`` sits below
``core``; ``colors``, ``colormaps`` and the package itself below that.

The imports that break the rule today, and the import cycles, are listed below
as known exceptions. The lists must match the tree exactly: a new crossing or a
new cycle fails, and so does one that has gone, so a fix also removes its entry
and the lists can only shrink. ``python tools/import_layers.py`` prints the
current state in the form used here.

The graph is read from the source, never by importing it. An import inside a
function counts as deferred; one under ``if TYPE_CHECKING:`` never runs and is
not counted; a module named in a string (the format registry's lazy readers,
the plugin loader) is not seen.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest


def _tool():
    """``tools/import_layers.py``, loaded by path (``tools`` is no package)."""
    name = "_mfv_import_layers"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "tools" / "import_layers.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # a dataclass looks its module up there
    spec.loader.exec_module(module)
    return module


layers = _tool()


#: Imports that cross the layers the wrong way, as they stand on 9175afb.
KNOWN_CROSSINGS = frozenset({
    'analysis.local_density -> ui.modeless (deferred)',
    'analysis.local_density -> ui.plot_format (import time)',
    'analysis.local_density -> ui.qt_lifecycle (deferred)',
    'analysis.localization_precision -> ui.modeless (deferred)',
    'analysis.localization_precision -> ui.qt_lifecycle (deferred)',
    'analysis.method_text -> plugins (deferred)',
    'analysis.method_text -> plugins.method (deferred)',
    'analysis.trace_analysis -> ui.modeless (deferred)',
    'analysis.trace_analysis -> ui.qt_lifecycle (deferred)',
    'analysis.trace_analysis -> ui.text_select (deferred)',
    'analysis.trace_analysis -> ui.z_scaling_widgets (deferred)',
    'core.app_state -> scripting (deferred)',
    'core.confocal_mapping -> msr.acquisition_roi (import time)',
    'core.confocal_mapping -> msr.mfxdta (import time)',
    'core.mbm_file -> msr.legacy_mbm (import time)',
    'core.minflux_zarr -> msr (deferred)',
    'core.minflux_zarr -> msr.io (deferred)',
    'core.minflux_zarr -> msr.zarr2 (deferred)',
    'core.obf_image_source -> msr (import time)',
    'core.obf_image_source -> msr.acquisition_roi (deferred)',
    'core.obf_image_source -> msr.mfxdta (deferred)',
    'core.overlay -> msr.alignment (deferred)',
    'core.post_load -> analysis.local_density (deferred)',
    'core.post_load -> analysis.localization_precision (deferred)',
    'core.post_load -> analysis.trace_analysis (deferred)',
    'core.recorder -> api (deferred)',
    'core.save -> msr.writer (deferred)',
    'core.spatial_grid -> ui.render_config (import time)',
    'utils.paraview_export -> core.dataset (import time)',
})

#: Import cycles that run at import time.
KNOWN_IMPORT_TIME_CYCLES = frozenset({
    ('msr', 'msr.io'),
})

#: Import cycles once deferred imports are counted too.
KNOWN_CYCLES = frozenset({
    ('analysis.method_text', 'plugins', 'plugins.data_simulator',
     'plugins.drift_correction', 'plugins.generate_method_text',
     'plugins.hlyb_pair_analysis', 'plugins.macro_recorder',
     'plugins.msr_reader', 'plugins.paraview', 'plugins.script_editor',
     'plugins.spatial_line_pattern', 'plugins.trace_viewer',
     'plugins.trace_viewer.trace_viewer_window'),
    ('core.acquisition_time', 'core.dataset', 'core.loader',
     'core.minflux_zarr', 'core.obf_image_source', 'core.overlay_save',
     'core.save', 'msr', 'msr.export', 'msr.io', 'msr.mfxdta',
     'msr.msr_parser', 'msr.viewer_state', 'msr.writer'),
    ('core.processing_journal', 'core.recorder'),
    ('core.roi', 'core.roi_logic'),
    ('ui.filter_dialog', 'ui.histogram_window'),
})


def _report(kind: str, found: set, known: frozenset) -> str:
    new = sorted(found - known)
    gone = sorted(known - found)
    lines = []
    if new:
        lines.append(f"New {kind} -- not allowed. Fix the import; list it here "
                     "only if it is deliberate, with the reason:")
        lines += [f"    {item!r}," for item in new]
    if gone:
        lines.append(f"{kind.capitalize()} that no longer exist -- remove them "
                     "from the list (it may only shrink):")
        lines += [f"    {item!r}," for item in gone]
    return "\n".join(lines)


# --------------------------------------------------------------- the analyser
def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "pkg"
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    return root


def _edges(root: Path) -> set[tuple[str, str, str]]:
    return {(e.source, e.target, e.kind) for e in layers.scan(root, "pkg")}


def test_relative_and_absolute_imports_resolve_to_modules(tmp_path):
    root = _package(tmp_path, {
        "__init__.py": "",
        "core/__init__.py": "",
        "core/a.py": "from .b import f\nfrom . import c\n",
        "core/b.py": "def f(): pass\n",
        "core/c.py": "",
        "ui/__init__.py": "",
        "ui/w.py": "from ..core.a import x\nimport pkg.core.c\nfrom pkg import core\n",
    })
    edges = _edges(root)
    assert ("pkg.core.a", "pkg.core.b", "import time") in edges
    assert ("pkg.core.a", "pkg.core.c", "import time") in edges
    assert ("pkg.core.a", "pkg.core", "import time") in edges     # `from . import`
    assert ("pkg.ui.w", "pkg.core.a", "import time") in edges
    assert ("pkg.ui.w", "pkg.core.c", "import time") in edges
    assert ("pkg.ui.w", "pkg.core", "import time") in edges
    assert not any(target.endswith((".f", ".x")) for _, target, _ in edges)


def test_where_an_import_sits_decides_its_kind(tmp_path):
    root = _package(tmp_path, {
        "__init__.py": "",
        "a.py": "",
        "b.py": "",
        "c.py": "",
        "d.py": "",
        "m.py": """
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                from . import a
            else:
                from . import d

            class K:
                from . import b

                def method(self):
                    from . import c
        """,
    })
    edges = {(target, kind) for source, target, kind in _edges(root)
             if source == "pkg.m"}
    assert ("pkg.b", "import time") in edges          # a class body runs on import
    assert ("pkg.c", "deferred") in edges             # a function body does not
    assert ("pkg.d", "import time") in edges          # the else branch does run
    assert not any(target == "pkg.a" for target, _ in edges)   # typing only


def test_a_wrong_way_import_is_a_crossing_and_api_may_reach_ui_lazily(tmp_path):
    root = _package(tmp_path, {
        "__init__.py": "",
        "core/__init__.py": "",
        "core/x.py": "from ..ui import w\n",
        "ui/__init__.py": "",
        "ui/w.py": "from ..core import x\n",
        "api/__init__.py": "",
        "api/lazy.py": "def f():\n    from ..ui import w\n",
        "api/eager.py": "from ..ui import w\n",
    })
    found = layers.crossings(layers.scan(root, "pkg"), "pkg")
    assert found == {
        "core.x -> ui (import time)",
        "core.x -> ui.w (import time)",
        "api.eager -> ui (import time)",
        "api.eager -> ui.w (import time)",
    }


def test_cycles_are_found_and_deferred_ones_only_with_deferred_imports(tmp_path):
    root = _package(tmp_path, {
        "__init__.py": "",
        "a.py": "from . import b\n",
        "b.py": "from . import a\n",
        "c.py": "from . import d\n",
        "d.py": "def f():\n    from . import c\n",
    })
    edges = layers.scan(root, "pkg")
    assert layers.cycles(edges, "pkg", import_time_only=True) == {("a", "b")}
    assert layers.cycles(edges, "pkg") == {("a", "b"), ("c", "d")}


def test_a_name_outside_every_layer_is_reported(tmp_path):
    root = _package(tmp_path, {"__init__.py": "", "newthing/__init__.py": ""})
    assert layers.unplaced(layers.scan(root, "pkg"), root, "pkg") == {"newthing"}


# ------------------------------------------------------------------ the tree
@pytest.fixture(scope="module")
def tree():
    return layers.scan(layers.PACKAGE_DIR, layers.PACKAGE)


def test_every_part_of_the_package_has_a_layer(tree):
    missing = layers.unplaced(tree, layers.PACKAGE_DIR, layers.PACKAGE)
    assert not missing, (
        f"{sorted(missing)} belong to no layer: add them to LAYER_OF in "
        "tools/import_layers.py.")


def test_no_new_import_crosses_the_layers_the_wrong_way(tree):
    found = layers.crossings(tree, layers.PACKAGE)
    assert found == KNOWN_CROSSINGS, _report("crossings", found, KNOWN_CROSSINGS)


def test_no_new_import_time_cycle(tree):
    found = layers.cycles(tree, layers.PACKAGE, import_time_only=True)
    assert found == KNOWN_IMPORT_TIME_CYCLES, _report(
        "import-time cycles", found, KNOWN_IMPORT_TIME_CYCLES)


def test_no_new_cycle_through_deferred_imports(tree):
    found = layers.cycles(tree, layers.PACKAGE)
    assert found == KNOWN_CYCLES, _report("cycles", found, KNOWN_CYCLES)
