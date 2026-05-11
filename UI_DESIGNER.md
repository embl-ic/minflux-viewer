# Editing the UI with Qt Designer

Several windows in this project have their layout defined in Qt Designer
`.ui` files instead of being hand-built in Python. This document
describes the workflow for editing them.

## Files

```
resources/ui/                    ← human-edited .ui (XML) files
    main_window.ui               ← menus, toolbar, status bar
    data_window.ui               ← per-dataset info card (reference)
    filter_dialog.ui             ← filter dialog shell (reference)
    dataset_manager.ui           ← dataset manager shell (reference)

minflux_viewer/ui/generated/     ← AUTO-GENERATED — do not hand-edit
    main_window_ui.py            ← produced by pyuic6 from main_window.ui
    data_window_ui.py
    filter_dialog_ui.py
    dataset_manager_ui.py

tools/
    compile_ui.py                ← script that runs pyuic6 on every .ui
```

## Quick edit loop

1. **Open a `.ui` file in Qt Designer**:
   ```powershell
   # Qt Designer ships with PyQt6-tools
   poetry run pyqt6-tools designer  resources\ui\main_window.ui
   ```
   Or just double-click the file if you have a standalone Qt Designer
   install.

2. **Make your edits** visually — add widgets, tweak layouts, change
   spacing, rename objects, wire up additional actions, etc.

3. **Save** the `.ui` file (Ctrl+S in Designer).

4. **Regenerate the Python bindings**:
   ```powershell
   poetry run python tools/compile_ui.py
   ```
   This runs `pyuic6` on each `.ui` file and writes `_ui.py` modules
   into `minflux_viewer/ui/generated/`.

5. **Run the app** — your changes appear immediately:
   ```powershell
   poetry run minflux-viewer
   ```

### Auto-rebuild during development

```powershell
poetry run python tools/compile_ui.py --watch
```

Requires `watchdog` (`pip install watchdog`). The script watches
`resources/ui/` and recompiles any `.ui` file the moment you save in
Designer.

### CI / pre-commit check

```powershell
poetry run python tools/compile_ui.py --check
```

Exits non-zero if any `_ui.py` module is older than its `.ui` source.
Useful as a pre-commit hook to stop drift between designer and code.

## What is — and isn't — in the .ui file

**In the `.ui`** (edit in Designer):
- Widget tree (menus, toolbars, layouts, labels, buttons, …)
- Widget properties (text, tooltips, size policies, style sheets, …)
- Keyboard shortcuts on `QAction`s
- Layout margins and spacing
- Default geometry / window title

**NOT in the `.ui`** (lives in Python):
- Signal/slot connections — wired manually in `_connect_actions()` so
  Python logic stays close to the handler functions.
- Dynamic content — table rows, tree nodes, combo box items that
  depend on loaded data.
- Custom widget subclasses (e.g. `_WelcomeWidget` with its own
  drag-drop forwarding). Designer can represent these as
  "promoted widgets" but we kept them in code to avoid coupling.
- `pyqtgraph` plot widgets (`PlotWidget`, `ImageView`, …). These are
  constructed at runtime in the tool-window modules
  (`scatter_window.py`, `render_window.py`, …).

## How the generated code is used

The pattern is the standard Qt one: a `Ui_*` class supplies
`setupUi()` which populates a `QMainWindow` / `QDialog` / `QWidget`,
and the Python class combines `setupUi` with the behaviour:

```python
# minflux_viewer/ui/main_window.py
from .generated.main_window_ui import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self, state):
        super().__init__()
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        # Connect actions created by setupUi
        self._ui.actionOpen.triggered.connect(self._open_dialog)
        # ...
```

Accessing widgets in Python uses `self._ui.<objectName>` where
`objectName` is what you set in Designer's property panel.

## Tool windows NOT in Designer

These are not in `resources/ui/` because their main widget is a
pyqtgraph plot (which Designer cannot preview or edit):

- `histogram_window.py` — PlotWidget + draggable filter lines
- `scatter_window.py` — ScatterPlotItem
- `attribute_window.py` — PlotWidget
- `render_window.py` — ImageView with Z slider
- `log_window.py` — QPlainTextEdit with 3 buttons (too simple)
- `msr_import_dialog.py` — QThread-driven parse result tree

If you add a new window whose layout is mostly static widgets, create
a new `.ui` file in `resources/ui/` and it will be picked up
automatically by `compile_ui.py`.

## Troubleshooting

**`pyuic6: command not found`**
PyQt6 was not installed via Poetry. Run `poetry install`.

**Widget missing after editing the .ui**
Did you re-run `tools/compile_ui.py`? The Python code imports from
`minflux_viewer/ui/generated/*_ui.py` which only updates when you
explicitly regenerate.

**A property I set in Designer has no effect**
Check the generated `*_ui.py` file to see if the property made it
through. Some properties only apply to specific widget subclasses;
others (like signals/slots set in Designer's Signal/Slot Editor) are
intentionally ignored — we wire signals in Python instead.
