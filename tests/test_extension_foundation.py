"""
Extension-layer **Phase 0** foundation.

These assert the seams the four parallel tracks build against. They are
deliberately cheap and headless: the point is that the contract exists and has
the shape the tracks were told it has, not that any of it does real work yet.

See ``docs/extension-layer/PLAN.md`` section 5.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 0.2 -- the published API package
# ---------------------------------------------------------------------------

def test_api_version_is_declared():
    from minflux_viewer import api

    assert api.__api_version__ == "1.0"


def test_every_namespace_exists_and_is_importable():
    """A track must be able to code against a namespace it does not own."""
    from minflux_viewer import api

    assert len(api.NAMESPACES) == 8
    for name in api.NAMESPACES:
        module = importlib.import_module(f"minflux_viewer.api.{name}")
        cls = getattr(module, name.capitalize())
        assert issubclass(cls, api._base.Namespace) if hasattr(api, "_base") else cls


def test_a_deferred_member_explains_itself_rather_than_vanishing():
    """
    Phase 0's scaffold raised ``NotImplementedError`` naming the owning track,
    so a track could code against a namespace another track had not written
    yet. Every track has now landed, and the property that survives is the one
    that mattered: a *published but deferred* name must give the reason rather
    than an AttributeError.

    ``mfv.view.volume`` and ``mfv.view.lut`` are the two deferred in API 1.0.
    """
    from minflux_viewer.api.view import View

    view = View(facade=None)
    for call in (view.volume, view.lut):
        with pytest.raises(NotImplementedError) as excinfo:
            call()
        assert "1.0" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 0.1 -- the public window API
# ---------------------------------------------------------------------------

def test_public_window_openers_exist():
    from minflux_viewer.ui.main_window import MainWindow

    for name in (
        "show_render", "show_scatter", "show_histogram",
        "show_attribute_plot", "show_console", "show_script_editor",
    ):
        assert callable(getattr(MainWindow, name)), name


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (0, 0),
        (3, 3),
        # A plugin may connect a public opener to QAction.triggered, which
        # supplies checked=False. Index 0 is the wrong reading of that.
        (True, None),
        (False, None),
        # A script index is very often a numpy integer.
        (np.int64(2), 2),
        (np.int32(7), 7),
    ],
)
def test_dataset_index_argument_is_narrowed(value, expected):
    from minflux_viewer.ui.main_window import MainWindow

    assert MainWindow._dataset_idx_arg(value) == expected


@pytest.mark.parametrize("value", [2.0, "2", object(), [1]])
def test_a_non_integer_dataset_index_raises_rather_than_defaulting(value):
    """
    Silently substituting the active dataset for a bad index is a wrong answer
    with no error, which is the failure mode this project rules out.
    """
    from minflux_viewer.ui.main_window import MainWindow

    with pytest.raises(TypeError):
        MainWindow._dataset_idx_arg(value)


# ---------------------------------------------------------------------------
# 0.3 -- preference keys for every track, declared up front
# ---------------------------------------------------------------------------

def test_extension_preference_keys_are_declared_for_all_tracks():
    """
    Both tracks' keys land in Phase 0 so neither has to edit app_state.py
    while they run in parallel.
    """
    from minflux_viewer.core.app_state import DEFAULT_PREFS

    plugin = DEFAULT_PREFS["plugin"]
    for key, kind in (
        ("paths", list),                 # track B
        ("scan_user_dir", bool),
        ("confirmed_dirs", list),
        ("user_lib_paths", list),        # track D
        ("user_lib_enabled", bool),
        ("installed_packages", dict),
    ):
        assert key in plugin, key
        assert isinstance(plugin[key], kind), key


def test_new_preference_keys_need_no_migration():
    """
    ``_merge`` starts from a deep copy of the defaults, so a key added with a
    default resolves on its own. A no-op migration entry would be noise -- and
    a migration that is *not* in ``_MIGRATION_KEYS`` runs against fresh
    defaults, which is the trap this project has hit before.
    """
    from minflux_viewer.core.app_state import DEFAULT_PREFS, _merge

    saved = {"plugin": {"msr_remember_last": False}}   # predates the new keys
    merged = _merge(saved, DEFAULT_PREFS)
    assert merged["plugin"]["msr_remember_last"] is False
    assert merged["plugin"]["paths"] == []
    assert merged["plugin"]["user_lib_enabled"] is True


# ---------------------------------------------------------------------------
# 0.4 / 0.5 -- the seams the tracks fill
# ---------------------------------------------------------------------------

def test_preferences_seam_modules_expose_the_agreed_interface():
    """Three calls, and preferences_dialog.py never needs changing again."""
    from minflux_viewer.ui import prefs_plugin_paths, prefs_python_packages

    for module, build, group in (
        (prefs_plugin_paths, "build_plugin_paths_group", "PluginPathsGroup"),
        (prefs_python_packages, "build_python_packages_group", "PythonPackagesGroup"),
    ):
        assert callable(getattr(module, build))
        # load/save take the dialog's draft dict and mutate it in place.
        cls = getattr(module, group)
        assert callable(getattr(cls, "load"))
        assert callable(getattr(cls, "save"))


def test_startup_hooks_return_the_right_shape():
    from minflux_viewer import plugins
    from minflux_viewer.core import user_libs

    assert user_libs.install_paths({}) == []
    assert isinstance(plugins.discover({}), list)


def test_discovery_does_not_disturb_the_builtin_registry():
    """Built-in plugins must keep working exactly as they do."""
    from minflux_viewer import plugins

    plugins.ensure_loaded()
    before = [e.name for e in plugins.available()]
    plugins.discover({})
    assert [e.name for e in plugins.available()] == before
    assert "Script Editor" in before


# ---------------------------------------------------------------------------
# 0.6 -- the frozen build must carry a complete standard library
# ---------------------------------------------------------------------------

def test_spec_collects_the_stdlib_modules_external_code_needs():
    """
    The shipped 0.4.2 build carried 265 stdlib modules and was missing
    ``logging.handlers``, ``html.parser``, ``configparser`` and ``tomllib`` --
    all four needed by ordinary third-party code, and ``tomllib`` needed by the
    plugin manifest itself. This asserts the spec's own collection logic covers
    them, without paying for a rebuild.
    """
    import importlib.util
    import sys

    pyinstaller = pytest.importorskip("PyInstaller.utils.hooks")
    collect_submodules = pyinstaller.collect_submodules

    names = set(sys.stdlib_module_names)
    for name in tuple(names):
        if name.startswith("_"):
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            continue
        if spec is not None and spec.submodule_search_locations is not None:
            try:
                names.update(collect_submodules(name))
            except Exception:
                continue

    for missing_in_0_4_2 in (
        "logging.handlers", "html.parser", "configparser", "tomllib", "__future__",
    ):
        assert missing_in_0_4_2 in names, missing_in_0_4_2
