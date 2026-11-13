# Developer guide

How the code is organised, how it is checked, how it is built, and how these
documents are kept current. The [README](../../README.md#for-developers) has the
design principles and the data model in brief; this page covers the working
practice.

## Set up

Python 3.10, 3.11 or 3.12. The dependencies are locked with Poetry; CI uses
Poetry 2.3.2.

```bash
git clone https://github.com/embl-ic/minflux-viewer.git
cd minflux-viewer
poetry sync --with build          # the locked environment, including PyInstaller
poetry run python -m minflux_viewer
```

`pip install -e .` in a virtual environment also works for running the
application (see the README), but the tests and the build expect the locked
Poetry environment.

## Code layout and layers

```
minflux_viewer/
  colors.py, colormaps.py   base
  utils/                    small helpers
  core/                     data model, loaders, save/export, application state
  msr/                      the .msr reader and writer
  analysis/                 scientific methods
  api/, scripting.py        the public mfv API
  ui/, plugins/             windows, dialogs, menus; built-in plugins
tests/                      pytest suite
tools/                      quality gates and generators
docs/                       this documentation
```

Each layer may import itself and the layers above it in this list, never the
ones below: `core` knows nothing of `analysis`, and `utils`, `core`, `msr` and
`analysis` import no Qt widgets. `api` may reach `ui` and `plugins` only inside a function, because
it acts on a running application. `tests/test_import_layers.py` checks this from
the source; known exceptions are listed there and the list may only shrink.
`python tools/import_layers.py` prints the current state.

Two rules shape most of the code:

- **One internal form.** Every file format is converted to one dataset
  representation on load, so windows and analyses never branch on the source
  format.
- **Data, derived values and view state are separate.** The canonical
  localization store is never modified by filters, ROIs or display settings;
  operations that change data create a new dataset.

## Tests

```bash
poetry run pytest -m "not qt"                    # pure tests, one process
poetry run python tools/run_qt_tests.py -j 4     # Qt tests, one module per process
```

Tests that need a `QApplication` request pytest-qt's `qapp` or `qtbot` and are
marked `qt` automatically; a guard fails any other test that starts Qt. The Qt
tests run one module per process because a native crash in one module must not
take the others down; `--strict` (as in CI) also fails a module that crashed
once and passed on retry. Tests that need the real sample data skip when it is
absent.

CI (`.github/workflows/ci.yml`) runs on every push: the pure and Qt suites on
Windows with Python 3.10, 3.11 and 3.12, the pure suite on Linux and macOS, and
the lint gates. Runners use UTC and UTF-8 mode; to reproduce locally run the
pure suite with `TZ=UTC0 PYTHONUTF8=1`.

## Lint gates

```bash
poetry run python tools/lint_ratchet.py          # Ruff and mypy
poetry run python tools/exception_ratchet.py
git diff --check "$(git hash-object -t tree /dev/null)" HEAD
```

The ratchets compare the findings per file and rule with a committed baseline
(`tools/lint_baseline.json`): a change may remove findings, never add them
unnoticed. Check the exit code, not only the printout. The whitespace check runs
over every committed file, so run it on the commit, including new files.

## Build the application

PyInstaller is pinned in the `build` dependency group. Build on the platform
you ship for — there is no cross-building.

```powershell
poetry sync --with build
.\.venv\Scripts\python.exe tools\vendor_pip.py      # generated build input (resources/pip)
.\.venv\Scripts\python.exe -m PyInstaller --clean --noconfirm minflux_viewer.spec
```

The Windows application is written to `dist\minflux_viewer\`; on macOS the spec
produces `dist/MINFLUX Data Viewer.app`. Always run the built application once — a
successful build is not a successful start.

A release bumps the version in `pyproject.toml` and
`minflux_viewer/__init__.py` (the build reads it from there for the Windows
file properties) and adds a short entry at the top of `CHANGELOG.md`. Release
archives are named `MINFLUX-Viewer-<version>-win.zip` (and `-macOS…`,
`-linux…`): the in-app update check picks the archive whose name matches the
platform, and the Windows archive holds a single `minflux_viewer/` folder.

## Extending the application

- **Scripts and plugins** use the public `mfv` API, described in the
  [scripting and plugin API](../scripting-api.md). Per-project analysis belongs
  in a plugin rather than in the core.
- **A new menu command** gets an entry in
  `minflux_viewer/ui/common/command_meta.py` — its implementing file, search
  keywords, a one-line summary, and its parameters, inputs and outputs. The
  Command Finder, the macro recorder and the
  [command reference](../reference/commands.md) all read it.
- **The main window's menus and toolbar** are defined in a Qt Designer file; see
  [Editing the UI with Qt Designer](../ui-designer.md).
- **The application's dataset format** is specified in
  [MINFLUX Viewer Zarr v2](../zarr-v2-format.md).

## Documentation

`docs/` holds the published documentation, written in Markdown so it reads on
GitHub as it is. Pages describe the current code; a change that alters
behaviour updates the page that describes it.

The [command reference](../reference/commands.md) is generated from the
application:

```bash
poetry run python tools/build_command_reference.py          # rewrite the page
poetry run python tools/build_command_reference.py --check  # is it current?
```

`tests/test_docs_command_reference.py` fails when the page and the menus or
`command_meta.py` disagree, so a renamed command or a new parameter cannot leave
the reference stale.
