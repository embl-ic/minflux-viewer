"""
Extension-layer **track B** -- discovery of user plugins from folders outside
the package.

The properties that matter and are easy to lose:

* two customers' plugins may both contain a ``utils.py``;
* a broken plugin must not stop the others, or the built-ins, reaching the menu;
* nothing is imported while the menu is built;
* code from a folder the user has not accepted does not run.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def _tier1(root, name: str, body: str = "") -> None:
    _write(root / f"{name}.py", body or "def run(ctx):\n    return 'ran'\n")


def _tier2(root, folder: str, *, plugin_id: str, name: str = "Tool",
           menu_path: str = "", body: str | None = None,
           extra_manifest: str = "", module: str = "main") -> None:
    _write(root / folder / "plugin.toml", f"""
        [plugin]
        id   = "{plugin_id}"
        name = "{name}"
        version = "1.0.0"

        [menu]
        path  = "{menu_path}"
        entry = "{name}"

        [run]
        module   = "{module}"
        callable = "run"
        {extra_manifest}
    """)
    if body is not None:
        _write(root / folder / f"{module}.py", body)


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A prefs dict pointing discovery at a temp folder and nowhere else."""
    root = tmp_path / "plugins"
    root.mkdir()
    app_root = tmp_path / "app-plugins"
    app_root.mkdir()
    from minflux_viewer.plugins import loader
    monkeypatch.setattr(loader, "app_plugin_dir", lambda: app_root)
    prefs = {"plugin": {
        "paths": [str(root)],
        "scan_user_dir": False,
        "confirmed_dirs": [str(root)],
    }}
    return root, prefs


@pytest.fixture(autouse=True)
def _clean_registry():
    """
    Keep discovered entries and imported plugin modules out of other tests.

    ``ensure_loaded()`` first, deliberately: it is one-shot (``_LOADED``), so a
    snapshot taken before the built-ins have registered would be restored as a
    truncated registry that nothing can refill.
    """
    from minflux_viewer import plugins
    from minflux_viewer.plugins import loader

    plugins.ensure_loaded()
    before = list(plugins._REGISTRY)
    yield
    plugins._REGISTRY[:] = before
    loader.unload_plugin_modules()


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def test_manifest_reads_the_documented_shape(roots):
    from minflux_viewer.plugins.manifest import read_manifest

    root, _ = roots
    _tier2(root, "dimer", plugin_id="embl.hlyb_dimer", name="Dimer Distance",
           menu_path="Plugins > HlyB", body="def run(ctx): pass\n",
           extra_manifest='\n[requires]\nmfv_api = ">=1.0,<2.0"\n')
    manifest = read_manifest(root / "dimer")

    assert manifest.id == "embl.hlyb_dimer"
    assert manifest.name == "Dimer Distance"
    assert manifest.menu_path == "Plugins > HlyB"
    assert manifest.module == "main" and manifest.callable_name == "run"
    assert manifest.requires_mfv_api == ">=1.0,<2.0"


@pytest.mark.parametrize("toml_text, expected", [
    ("", "[plugin]"),
    ('[plugin]\nname = "x"\n', "id"),
    ('[plugin]\nid = "a.b"\n', "name"),
    ('[plugin]\nid = "Not Valid"\nname = "x"\n', "lowercase"),
    ('[plugin]\nid = "a"\nname = "x"\n[run]\nmodule = "sub/dir"\n', "module"),
    ('[plugin]\nid = "a"\nname = "x"\n[requires]\nmfv_api = "1.0"\n', "must start with"),
])
def test_a_malformed_manifest_explains_itself(roots, toml_text, expected):
    from minflux_viewer.plugins.manifest import ManifestError, read_manifest

    root, _ = roots
    _write(root / "bad" / "plugin.toml", toml_text)
    with pytest.raises(ManifestError) as excinfo:
        read_manifest(root / "bad")
    assert expected in str(excinfo.value)


def test_invalid_toml_is_reported_as_such(roots):
    from minflux_viewer.plugins.manifest import ManifestError, read_manifest

    root, _ = roots
    _write(root / "bad" / "plugin.toml", "this is not = = toml\n")
    with pytest.raises(ManifestError) as excinfo:
        read_manifest(root / "bad")
    assert "valid TOML" in str(excinfo.value)


@pytest.mark.parametrize("version, requirement, ok", [
    ("1.0", ">=1.0,<2.0", True),
    ("1.9", ">=1.0,<2.0", True),
    ("2.0", ">=1.0,<2.0", False),
    ("0.9", ">=1.0", False),
    ("1.0", "", True),
    ("1.2.3", "==1.2.3", True),
    ("1.2.3", "!=1.2.3", False),
    ("0.4.2", ">=0.5.0", False),
    ("0.5.0", ">=0.5.0", True),
])
def test_version_requirements(version, requirement, ok):
    from minflux_viewer.plugins.manifest import version_satisfies

    assert version_satisfies(version, requirement) is ok


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def test_tier1_single_file_becomes_one_entry_named_fiji_style(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier1(root, "My_Project_Tool")
    found = loader.scan_root(root)

    assert len(found) == 1
    assert found[0].label == "My Project Tool"
    assert found[0].tier == 1
    assert found[0].error == ""


def test_tier1_skips_private_and_non_python_files(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier1(root, "Good")
    _write(root / "_helper.py", "x = 1\n")
    _write(root / "notes.txt", "hello\n")
    assert [p.label for p in loader.scan_root(root)] == ["Good"]


def test_tier2_directory_uses_its_manifest(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier2(root, "dimer", plugin_id="embl.dimer", name="Dimer Distance",
           body="def run(ctx): pass\n")
    found = loader.scan_root(root)
    assert len(found) == 1 and found[0].tier == 2
    assert found[0].id == "embl.dimer"
    assert found[0].label == "Dimer Distance"


def test_a_nested_menu_path_is_kept_separate_from_the_leaf_label(roots):
    """
    ``Plugins`` itself is dropped -- a manifest naming it is not asking for a
    submenu called Plugins inside Plugins -- while every remaining component
    is passed to the real menu builder.
    """
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier2(root, "d", plugin_id="a.b", name="Dimer",
           menu_path="Plugins > HlyB", body="def run(ctx): pass\n")
    found = loader.scan_root(root)[0]
    assert found.label == "Dimer"
    assert found.menu_path == ("HlyB",)


def test_a_broken_plugin_is_listed_with_its_reason(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _write(root / "broken" / "plugin.toml", '[plugin]\nname = "no id"\n')
    _tier1(root, "Fine")

    found = {p.label: p for p in loader.scan_root(root)}
    assert set(found) == {"broken", "Fine"}
    assert "id" in found["broken"].error
    assert found["Fine"].error == ""


def test_a_manifest_without_its_module_file_says_which_file_is_missing(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier2(root, "d", plugin_id="a.b", name="Tool", body=None)   # no main.py
    found = loader.scan_root(root)[0]
    assert "main.py" in found.error


def test_a_plugin_needing_a_newer_api_is_refused_with_both_versions(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier2(root, "future", plugin_id="a.future", name="Future",
           body="def run(ctx): pass\n",
           extra_manifest='\n[requires]\nmfv_api = ">=99.0"\n')
    found = loader.scan_root(root)[0]
    assert "99.0" in found.error and "1.2" in found.error


def test_a_plugin_needing_a_missing_package_says_where_to_get_it(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier2(root, "needs", plugin_id="a.needs", name="Needs",
           body="def run(ctx): pass\n",
           extra_manifest='\n[requires]\npython = ["definitely_not_installed_xyz"]\n')
    found = loader.scan_root(root)[0]
    assert "definitely_not_installed_xyz" in found.error
    assert "Python packages" in found.error


def test_scanning_a_missing_root_is_not_an_error(tmp_path):
    from minflux_viewer.plugins import loader

    assert loader.scan_root(tmp_path / "nope") == []


def test_roots_are_deduplicated_and_the_user_dir_can_be_switched_off(tmp_path):
    from minflux_viewer.plugins import loader

    shared = str(tmp_path / "p")
    prefs = {"plugin": {"paths": [shared, shared], "scan_user_dir": False}}
    roots = loader.plugin_roots(prefs)
    assert loader.user_plugin_dir() not in roots
    assert sum(str(r) == shared for r in roots) == 1


def test_the_app_root_is_resolved_from_the_executable_when_frozen(monkeypatch, tmp_path):
    """
    ``_MEIPASS`` is the bundle's ``_internal/`` -- the wrong place to look for a
    user's plugins, and replaced on every update.
    """
    from minflux_viewer.plugins import loader

    exe = tmp_path / "app" / "minflux_viewer.exe"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "app" / "_internal"), raising=False)

    assert loader.app_plugin_dir() == exe.parent / "plugins"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def test_two_plugins_may_both_ship_a_utils_module(roots):
    """The reason plugins are imported under ``mfv_plugins.<id>``."""
    from minflux_viewer.plugins import loader

    root, _ = roots
    for folder, plugin_id, value in (("one", "cust.one", 1), ("two", "cust.two", 2)):
        _tier2(root, folder, plugin_id=plugin_id, name=folder,
               body="from .utils import VALUE\n\ndef run(ctx):\n    return VALUE\n")
        _write(root / folder / "utils.py", f"VALUE = {value}\n")

    results = {}
    for found in loader.scan_root(root):
        assert found.error == "", found.error
        results[found.id] = loader.load_entry_callable(found)(None)

    assert results == {"cust.one": 1, "cust.two": 2}


def test_a_plugin_body_is_not_imported_until_it_is_launched(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    marker = root / "imported.txt"
    _tier2(root, "lazy", plugin_id="a.lazy", name="Lazy", body=f"""
        from pathlib import Path
        Path(r"{marker}").write_text("imported")

        def run(ctx):
            return "ran"
    """)

    found = loader.scan_root(root)[0]
    assert not marker.exists(), "scanning must not import the plugin body"

    assert loader.load_entry_callable(found)(None) == "ran"
    assert marker.exists()


def test_a_plugin_without_its_entry_callable_says_what_to_define(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier2(root, "d", plugin_id="a.b", name="Tool", body="x = 1\n")
    found = loader.scan_root(root)[0]
    with pytest.raises(AttributeError) as excinfo:
        loader.load_entry_callable(found)
    assert "def run(ctx)" in str(excinfo.value)


def test_a_failed_import_does_not_leave_a_half_module_behind(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier2(root, "d", plugin_id="a.boom", name="Boom",
           body="raise RuntimeError('boom')\n")
    found = loader.scan_root(root)[0]
    with pytest.raises(RuntimeError):
        loader.load_entry_callable(found)
    assert loader.module_name_for("a.boom") not in sys.modules


def test_unload_lets_an_edited_plugin_be_picked_up_again(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier2(root, "d", plugin_id="a.b", name="Tool",
           body="def run(ctx):\n    return 'first'\n")
    found = loader.scan_root(root)[0]
    assert loader.load_entry_callable(found)(None) == "first"

    _write(root / "d" / "main.py", "def run(ctx):\n    return 'second'\n")
    assert loader.load_entry_callable(found)(None) == "first", "cached until unloaded"

    loader.unload_plugin_modules()
    found = loader.scan_root(root)[0]
    assert loader.load_entry_callable(found)(None) == "second"


# ---------------------------------------------------------------------------
# registration and the menu
# ---------------------------------------------------------------------------

def test_discover_registers_user_plugins_without_disturbing_builtins(roots):
    from minflux_viewer import plugins

    root, prefs = roots
    plugins.ensure_loaded()
    builtins = [e.name for e in plugins.available()]
    assert "Script Editor" in builtins

    _tier1(root, "User_Tool")
    added = plugins.discover(prefs)

    assert [e.name for e in added] == ["User Tool"]
    names = [e.name for e in plugins.available()]
    assert names[:len(builtins)] == builtins        # built-ins unchanged, in order
    assert "User Tool" in names


def test_discover_is_idempotent(roots):
    from minflux_viewer import plugins

    root, prefs = roots
    _tier1(root, "Once")
    plugins.discover(prefs)
    again = plugins.discover(prefs)
    assert again == []
    assert sum(e.name == "Once" for e in plugins.available()) == 1


def test_menu_builder_reuses_real_submenus_and_disables_broken_entries(
        qapp, monkeypatch):
    from types import SimpleNamespace

    from PyQt6.QtWidgets import QMainWindow, QMenu

    from minflux_viewer import plugins
    from minflux_viewer.core import user_libs
    from minflux_viewer.ui.main_window import MainWindow

    entries = [
        plugins.PluginEntry(
            name="First", tooltip="one", launch=lambda *_: None,
            menu_path=("Lab",), discovered=True,
        ),
        plugins.PluginEntry(
            name="Second", tooltip="two", launch=lambda *_: None,
            menu_path=("Lab",), discovered=True,
        ),
        plugins.PluginEntry(
            name="Broken", tooltip="missing dependency", launch=lambda *_: None,
            menu_path=("Lab", "Diagnostics"), error="missing dependency",
            discovered=True,
        ),
    ]
    monkeypatch.setattr(plugins, "ensure_loaded", lambda: None)
    monkeypatch.setattr(plugins, "discover", lambda _prefs: [])
    monkeypatch.setattr(plugins, "available", lambda: entries)
    monkeypatch.setattr(user_libs, "install_paths", lambda _prefs: [])

    window = QMainWindow()
    window._ui = SimpleNamespace(menuPlugins=QMenu("Plugins", window))
    window._state = SimpleNamespace(prefs={}, log=lambda *_args, **_kwargs: None)
    window._mark_action_ai_unapproved = lambda _action: None

    MainWindow._populate_plugins_menu(window)

    top = window._ui.menuPlugins.actions()
    assert len(top) == 1 and top[0].menu().title() == "Lab"
    lab = top[0].menu()
    leaves = {action.text(): action for action in lab.actions() if action.menu() is None}
    assert set(leaves) == {"First", "Second"}
    diagnostics = next(action.menu() for action in lab.actions() if action.menu())
    broken = diagnostics.actions()[0]
    assert broken.text() == "Broken"
    assert broken.isEnabled() is False


def test_rediscover_drops_only_discovered_entries(roots):
    from minflux_viewer import plugins

    root, prefs = roots
    plugins.ensure_loaded()
    _tier1(root, "Temporary")
    plugins.discover(prefs)
    assert any(e.name == "Temporary" for e in plugins.available())

    (root / "Temporary.py").unlink()
    plugins.rediscover(prefs)

    names = [e.name for e in plugins.available()]
    assert "Temporary" not in names
    assert "Script Editor" in names          # a built-in survived


def test_a_broken_plugin_reports_instead_of_running(roots):
    from minflux_viewer import plugins

    root, prefs = roots
    _write(root / "broken" / "plugin.toml", '[plugin]\nname = "no id"\n')
    entry = plugins.discover(prefs)[0]
    assert entry.error

    logged = []

    class _State:
        prefs = {"plugin": {"confirmed_dirs": [str(root)]}}

        def log(self, message, level="INFO"):
            logged.append((level, message))

    entry.launch(_State(), None)
    assert logged and logged[0][0] == "ERROR"
    assert "cannot run" in logged[0][1]


def test_a_plugin_failure_is_contained_and_reported(roots):
    from minflux_viewer import plugins

    root, prefs = roots
    _tier2(root, "d", plugin_id="a.b", name="Explodes",
           body="def run(ctx):\n    raise ValueError('bang')\n")
    entry = plugins.discover(prefs)[0]

    logged = []

    class _State:
        prefs = {"plugin": {"confirmed_dirs": [str(root)]}}
        mfv = object()

        def log(self, message, level="INFO"):
            logged.append((level, message))

    entry.launch(_State(), None)          # must not raise
    assert logged and "bang" in logged[0][1]


def test_a_plugin_receives_the_mfv_facade_as_its_context(roots):
    from minflux_viewer import plugins

    root, prefs = roots
    marker = root / "ctx.txt"
    _tier2(root, "d", plugin_id="a.b", name="Ctx", body=f"""
        from pathlib import Path

        def run(ctx):
            Path(r"{marker}").write_text(type(ctx).__name__)
    """)
    entry = plugins.discover(prefs)[0]

    sentinel = type("Facade", (), {})()

    class _State:
        prefs = {"plugin": {"confirmed_dirs": [str(root)]}}
        mfv = sentinel

        def log(self, message, level="INFO"):
            pass

    entry.launch(_State(), None)
    assert marker.read_text() == "Facade"


# ---------------------------------------------------------------------------
# trust
# ---------------------------------------------------------------------------

def test_an_unconfirmed_folder_is_recorded_once_accepted(tmp_path):
    from minflux_viewer.plugins import loader

    prefs = {"plugin": {"confirmed_dirs": []}}
    root = tmp_path / "p"
    assert loader.root_is_confirmed(prefs, root) is False

    loader.confirm_root(prefs, root)
    assert loader.root_is_confirmed(prefs, root) is True

    loader.confirm_root(prefs, root)                       # idempotent
    assert len(prefs["plugin"]["confirmed_dirs"]) == 1


def test_confirmation_matches_regardless_of_path_spelling(tmp_path):
    from minflux_viewer.plugins import loader

    root = tmp_path / "p"
    root.mkdir()
    prefs = {"plugin": {"confirmed_dirs": [str(root)]}}
    assert loader.root_is_confirmed(prefs, root / "." ) is True


def test_a_dotted_plugin_id_becomes_one_module_segment():
    """
    ``mfv_plugins.embl.tool`` would imply an intermediate ``mfv_plugins.embl``
    package that does not exist.
    """
    from minflux_viewer.plugins import loader

    assert loader.module_name_for("embl.hlyb_dimer") == "mfv_plugins.embl_hlyb_dimer"
    assert loader.module_name_for("a-b.c") == "mfv_plugins.a_b_c"


def test_an_absolute_import_of_a_sibling_is_refused_not_silently_shared(roots):
    """
    Documented limitation, asserted so it cannot regress into the collision the
    namespacing exists to prevent: a plugin must use a relative import for its
    own helpers.
    """
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier2(root, "d", plugin_id="a.b", name="Tool", body="""
        import utils

        def run(ctx):
            return utils.VALUE
    """)
    _write(root / "d" / "utils.py", "VALUE = 1\n")
    found = loader.scan_root(root)[0]
    with pytest.raises(ModuleNotFoundError):
        loader.load_entry_callable(found)


# ---------------------------------------------------------------------------
# end to end -- a real plugin against the real facade
# ---------------------------------------------------------------------------

def test_a_plugin_can_use_the_real_mfv_api(roots, qapp):
    """
    The whole point, exercised once: a file outside the package is discovered,
    registered, launched, and does real work through the published API.
    """
    import numpy as np

    from minflux_viewer import plugins
    from minflux_viewer.core.app_state import AppState
    from minflux_viewer.core.dataset import build_localization_dataset

    root, prefs = roots
    out = root / "result.txt"
    _tier2(root, "measure", plugin_id="embl.measure", name="Measure",
           menu_path="Plugins > Demo", body=f"""
        from pathlib import Path

        def run(ctx):
            ds = ctx.data.active()
            loc = ctx.data.loc(unit="nm")
            ctx.data.add_attr("radius_nm", (loc[:, 0] ** 2 + loc[:, 1] ** 2) ** 0.5)
            ctx.journal.record("analysis", "Measured radii", n=len(loc))
            Path(r"{out}").write_text(f"{{ds.name}}:{{len(loc)}}")
    """)

    state = AppState()
    state.prefs["plugin"] = prefs["plugin"]
    state.add_dataset(build_localization_dataset(
        name="demo",
        x_nm=np.arange(10.0), y_nm=np.arange(10.0), z_nm=np.zeros(10),
    ))

    entry = [e for e in plugins.discover(state.prefs) if e.name == "Measure"][0]
    assert entry.menu_path == ("Demo",)
    assert entry.error == ""
    entry.launch(state, None)

    assert out.read_text() == "demo:10"
    # the plugin's attribute is now a first-class, selectable attribute
    assert "radius_nm" in state.mfv.data.attr_names()
    # and its run reached the record the method text is generated from
    assert list(state.journal)[-1].details["n"] == 10


def test_two_plugins_may_share_a_leaf_label_under_different_submenus(roots):
    """
    Identity is the manifest id, not the display name. Two vendors publishing
    *Lab A ▸ Analyse* and *Lab B ▸ Analyse* must both appear; the second is
    disambiguated rather than silently dropped.
    """
    from minflux_viewer import plugins

    root, prefs = roots
    _tier2(root, "a", plugin_id="lab.a", name="Analyse", menu_path="Plugins > Lab A",
           body="def run(ctx): pass\n")
    _tier2(root, "b", plugin_id="lab.b", name="Analyse", menu_path="Plugins > Lab B",
           body="def run(ctx): pass\n")

    added = plugins.discover(prefs)
    assert len(added) == 2
    assert {e.plugin_id for e in added} == {"lab.a", "lab.b"}
    assert len({e.name for e in added}) == 2, "both must be reachable"
    assert {e.menu_path for e in added} == {("Lab A",), ("Lab B",)}


def test_rescanning_is_idempotent_on_plugin_id(roots):
    from minflux_viewer import plugins

    root, prefs = roots
    _tier2(root, "a", plugin_id="lab.a", name="Analyse", body="def run(ctx): pass\n")
    assert len(plugins.discover(prefs)) == 1
    assert plugins.discover(prefs) == []
    assert sum(e.plugin_id == "lab.a" for e in plugins.available()) == 1


@pytest.mark.parametrize("requirement, expected", [
    ("numpy", ""),                                   # installed, no clause
    ("numpy>=2", ""),                                # installed, clause met
    ("scipy >= 1.0", ""),                            # whitespace tolerated
    ("numpy[extra]>=2", ""),                         # extras are pip's business
    ("definitely_absent_xyz", "definitely_absent_xyz"),
])
def test_a_satisfied_or_missing_requirement(requirement, expected):
    from minflux_viewer.plugins.loader import _requirement_unmet

    assert _requirement_unmet(requirement) == expected


def test_a_version_clause_is_actually_checked_and_reports_what_is_installed():
    """
    Review finding: clauses used to be stripped, so `numpy>=99` silently
    passed. The message must name both what was asked for and what is there.
    """
    from minflux_viewer.plugins.loader import _requirement_unmet

    reason = _requirement_unmet("numpy>=99")
    assert ">=99" in reason and "installed:" in reason


def test_a_plugin_needing_a_newer_package_version_is_refused(roots):
    from minflux_viewer.plugins import loader

    root, _ = roots
    _tier2(root, "d", plugin_id="a.b", name="Tool", body="def run(ctx): pass\n",
           extra_manifest='\n[requires]\npython = ["numpy>=99"]\n')
    found = loader.scan_root(root)[0]
    assert "numpy" in found.error and "installed:" in found.error
