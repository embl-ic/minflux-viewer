"""
minflux_viewer.ui.figure_export
===============================
Export **what a view shows** as a figure.

Two kinds of output, each from one source so the formats inside a kind cannot
disagree:

* **Pixel** — PNG, TIFF, JPEG. One image is rendered (optionally at an integer
  multiple of the screen resolution: plot lines, text and markers are drawn
  again at that scale, an image layer is resampled) and written in the chosen
  format. The same pixels go into every format; only the encoding differs.
* **Vector** — SVG, PDF. The view is painted once into the vector device:
  axes, text, lines and markers stay vector, an image layer (a render, a
  density map) is embedded as the image it is on screen.

What is exported is exactly the view as displayed — colormap, brightness /
contrast, zoom, overlays, colorbar and legend — and nothing else: no buttons,
no dropdowns.

OpenGL content (a 3-D view, the GPU-drawn points of the Attribute Plot) has no
vector form; it is taken from the GPU's framebuffer, as a screenshot would, and
composited at its place in the stacking order — the Attribute Plot's GPU canvas
sits *behind* a transparent plot, so it must be painted first, not on top. A
3-D view is therefore offered as a pixel image only.

This is a picture of the view. Exporting the *data* as a multi-page OME-TIFF
(Z planes, channels, voxel size) is the render view's separate *Export to TIFF*.

File names: the suggestion is ``<dataset>_<plot type>_<view>`` (e.g.
``run_260624-173440_minflux_loc_scatter_XY.pdf``), in the folder last exported
to. A name already taken gets the next free counter — ``<name>(_01).pdf``,
``<name>(_02).pdf`` — so repeated exports from one view never overwrite each
other; a view also remembers a name the user typed and continues from it.
Writing over an existing file is always confirmed first.
"""

from __future__ import annotations

import re
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPoint, QRect, QRectF, QSize, QSizeF
from PyQt6.QtGui import QColor, QImage, QPainter, QPageSize, QPdfWriter, QRegion
from PyQt6.QtWidgets import QWidget

PIXEL_FORMATS = {"png": ("PNG image", ".png"),
                 "tif": ("TIFF image", ".tif"),
                 "jpg": ("JPEG image", ".jpg")}
VECTOR_FORMATS = {"svg": ("SVG vector figure", ".svg"),
                  "pdf": ("PDF vector figure", ".pdf")}
FORMATS = {**PIXEL_FORMATS, **VECTOR_FORMATS}
MAX_SCALE = 8

#: Extensions that mark a figure file, so a dot inside a name is not mistaken
#: for one.
_FIGURE_EXTS = {ext for _label, ext in FORMATS.values()} | {".tiff", ".jpeg"}
#: Data-file extensions dropped when a dataset name becomes part of a file name.
_DATA_EXTS = (".msr", ".mat", ".npy", ".npz", ".json", ".csv", ".zarr", ".zip",
              ".h5", ".hdf5", ".tif", ".tiff")
#: The repeat counter at the end of a stem: ``name(_01)``.
_COUNTER_RE = re.compile(r"\(_(\d+)\)$")


@dataclass
class FigureTile:
    widget: QWidget
    x: int
    y: int


@dataclass
class Figure:
    """What to export: widgets placed on one canvas (logical pixels)."""
    tiles: list
    width: int
    height: int
    background: QColor
    name: str = "figure"
    is_3d: bool = False            # a 3-D view: pixel output only
    # What the suggested file name is made of; see suggested_stem().
    dataset: str = ""              # the dataset's name as the Dataset Manager shows it
    kind: str = ""                 # plot type: "render_view", "loc_scatter", …
    view: str = ""                 # view mode: "XY", "XZ", "Ortho-View", "3D", …
    folder_hint: str = ""          # where the dataset lives, when nothing was exported yet

    @property
    def has_opengl(self) -> bool:
        return any(_opengl_widgets(t.widget) for t in self.tiles)

    @property
    def vector_ok(self) -> bool:
        return not self.is_3d


def _is_opengl(widget) -> bool:
    try:
        from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    except Exception:                                   # pragma: no cover - no GL module
        return False
    return isinstance(widget, QOpenGLWidget)


def _opengl_widgets(root: QWidget) -> list:
    found = [root] if _is_opengl(root) and root.isVisible() else []
    for child in root.findChildren(QWidget):
        if _is_opengl(child) and child.isVisible():
            found.append(child)
    return found


def figure_of(widget: QWidget, *, name: str = "figure", background=None,
              is_3d: bool = False, **identity) -> Figure:
    """A figure of one widget (and everything drawn inside it).

    *identity*: ``dataset``, ``kind``, ``view``, ``folder_hint`` — see :class:`Figure`.
    """
    bg = QColor(background) if background is not None else widget.palette().window().color()
    return Figure([FigureTile(widget, 0, 0)], widget.width(), widget.height(), bg,
                  name=name, is_3d=is_3d, **identity)


def ortho_figure(xy: QWidget, yz: QWidget, xz: QWidget, *, name: str = "figure",
                 background=None, **identity) -> Figure:
    """The three orthogonal panes stacked tightly: XY, YZ to its right, XZ below."""
    w = xy.width() + yz.width()
    h = xy.height() + xz.height()
    bg = QColor(background) if background is not None else xy.palette().window().color()
    tiles = [FigureTile(xy, 0, 0), FigureTile(yz, xy.width(), 0),
             FigureTile(xz, 0, xy.height())]
    return Figure(tiles, w, h, bg, name=name, **identity)


def dataset_identity(state, idx) -> dict:
    """``dataset`` and ``folder_hint`` of dataset *idx*, for a view's figure."""
    try:
        ds = state.datasets[int(idx)]
    except (AttributeError, IndexError, TypeError, ValueError):
        return {}
    out = {"dataset": str(getattr(ds, "name", "") or "")}
    try:
        from ..core.dataset import dataset_source_file
        src = dataset_source_file(ds)
        if src is not None:
            out["folder_hint"] = str(src.parent)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# File names
# ---------------------------------------------------------------------------

def _safe_part(text) -> str:
    """One file-name component: characters no file system accepts, and blanks,
    become ``_``; runs collapse."""
    out = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', "_", str(text or ""))
    return re.sub(r"_+", "_", out).strip("._ ")


def dataset_label(name: str) -> str:
    """A dataset name as a file-name component.

    ``"run.msr | 260624-173440_minflux"`` → ``"run_260624-173440_minflux"``: the
    source file's extension goes, the parts are joined.
    """
    parts = [p.strip() for p in str(name or "").split("|") if p.strip()]
    if not parts:
        return ""
    first = parts[0]
    stripped = True
    while stripped:                                   # e.g. ".zarr.zip"
        stripped = False
        for ext in _DATA_EXTS:
            if first.lower().endswith(ext) and len(first) > len(ext):
                first, stripped = first[: -len(ext)], True
    parts[0] = first
    return _safe_part("_".join(parts))


def suggested_stem(figure: Figure) -> str:
    """``<dataset>_<plot type>_<view>`` — whichever parts the figure knows."""
    lead = dataset_label(figure.dataset) or _safe_part(figure.name) or "figure"
    parts = [lead] + [_safe_part(p) for p in (figure.kind, figure.view)]
    return "_".join(p for p in parts if p)


def base_stem(stem: str) -> str:
    """*stem* without its repeat counter: ``name(_03)`` → ``name``."""
    return _COUNTER_RE.sub("", str(stem))


def next_free_path(folder, stem: str, ext: str) -> Path:
    """``<stem><ext>`` in *folder*, or the first free ``<stem>(_NN)<ext>``."""
    folder = Path(folder)
    base = base_stem(stem) or "figure"
    candidate = folder / f"{base}{ext}"
    i = 0
    while candidate.exists():
        i += 1
        candidate = folder / f"{base}(_{i:02d}){ext}"
    return candidate


def with_extension(path, ext: str) -> Path:
    """*path* ending in *ext*: a figure extension is replaced, anything else —
    including a dot inside the name — is kept and *ext* appended."""
    p = Path(path)
    if p.suffix.lower() in _FIGURE_EXTS:
        return p.with_suffix(ext)
    return p.with_name(p.name + ext)


# ---------------------------------------------------------------------------
# Painting
# ---------------------------------------------------------------------------

def _paint_tree(painter: QPainter, widget: QWidget, root: QWidget) -> None:
    """Paint *widget* and its visible descendants in stacking order, each by
    itself, substituting the framebuffer for OpenGL widgets."""
    if not widget.isVisible():
        return
    off = widget.mapTo(root, QPoint(0, 0)) if widget is not root else QPoint(0, 0)
    if _is_opengl(widget):
        image = widget.grabFramebuffer()
        painter.drawImage(QRectF(off.x(), off.y(), widget.width(), widget.height()), image)
        return
    # QWidget.render, explicitly: a QGraphicsView (every pyqtgraph plot) has its
    # own render() that draws the SCENE and shadows the widget one.
    QWidget.render(widget, painter, off, QRegion(QRect(0, 0, widget.width(), widget.height())),
                   QWidget.RenderFlag(0))
    for child in widget.children():
        if isinstance(child, QWidget) and not child.isWindow():
            _paint_tree(painter, child, root)


@contextmanager
def _pyqtgraph_export_mode(figure: Figure, scale: float):
    """Put every pyqtgraph item of the figure in export mode while it is painted.

    A scatter plot normally stamps each marker from a cached pixmap, which a
    vector device stores as one embedded bitmap *per marker* (a 32 k-point
    scatter made a 127 MB SVG) and which a scaled image would blur. In export
    mode pyqtgraph draws every marker as a vector symbol at
    ``resolutionScale`` — the same mechanism its own exporters use.
    """
    from PyQt6.QtWidgets import QGraphicsView

    items = []
    for tile in figure.tiles:
        views = ([tile.widget] if isinstance(tile.widget, QGraphicsView) else []) +             tile.widget.findChildren(QGraphicsView)
        for view in views:
            scene = view.scene()
            if scene is None:
                continue
            for item in scene.items():
                if hasattr(item, "setExportMode"):
                    items.append(item)
    opts = {"antialias": True, "resolutionScale": float(scale)}
    for item in items:
        try:
            item.setExportMode(True, dict(opts))
        except Exception:
            pass
    try:
        yield
    finally:
        for item in items:
            try:
                item.setExportMode(False)
            except Exception:
                pass


def paint_figure(painter: QPainter, figure: Figure) -> None:
    painter.fillRect(QRectF(0, 0, figure.width, figure.height), figure.background)
    for tile in figure.tiles:
        painter.save()
        painter.translate(tile.x, tile.y)
        painter.setClipRect(QRectF(0, 0, tile.widget.width(), tile.widget.height()))
        _paint_tree(painter, tile.widget, tile.widget)
        painter.restore()


def render_pixels(figure: Figure, scale: int = 1) -> QImage:
    """The figure as one image, ``scale`` times the screen resolution."""
    scale = int(max(1, min(MAX_SCALE, scale)))
    image = QImage(QSize(figure.width * scale, figure.height * scale),
                   QImage.Format.Format_ARGB32)
    image.fill(figure.background)
    painter = QPainter(image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.scale(scale, scale)
        # At 1x the image is the screen, pixel for pixel; larger, markers are
        # redrawn as symbols at that size rather than enlarged pixmaps.
        with (_pyqtgraph_export_mode(figure, scale) if scale > 1 else nullcontext()):
            paint_figure(painter, figure)
    finally:
        painter.end()
    return image


def _write_tiff(image: QImage, path: Path, scale: int) -> None:
    import tifffile

    rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
    h, w = rgba.height(), rgba.width()
    ptr = rgba.constBits()
    ptr.setsize(rgba.sizeInBytes())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, rgba.bytesPerLine())[:, :w * 4]
    arr = arr.reshape(h, w, 4)
    rgb = np.ascontiguousarray(arr[:, :, :3])
    dpi = 96.0 * scale
    tifffile.imwrite(str(path), rgb, photometric="rgb", resolution=(dpi, dpi),
                     resolutionunit="INCH", compression="zlib")


def save_figure(figure: Figure, path, fmt: str, *, scale: int = 1,
                jpeg_quality: int = 95) -> dict:
    """Write *figure* to *path* as *fmt*; returns what was written."""
    fmt = str(fmt).lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown figure format {fmt!r}")
    path = Path(path)
    if not path.parent.is_dir():
        raise OSError(f"The folder {path.parent} does not exist.")
    if fmt in VECTOR_FORMATS and not figure.vector_ok:
        raise ValueError("A 3-D view is a rendered image; export it as a pixel image.")
    if fmt in PIXEL_FORMATS:
        image = render_pixels(figure, scale)
        if fmt == "tif":
            _write_tiff(image, path, scale)
        else:
            out = image
            if fmt == "jpg":
                out = image.convertToFormat(QImage.Format.Format_RGB32)
            ok = out.save(str(path), "PNG" if fmt == "png" else "JPG",
                          int(jpeg_quality) if fmt == "jpg" else -1)
            if not ok:
                raise OSError(f"could not write {path}")
        return {"format": fmt, "width": image.width(), "height": image.height(),
                "path": str(path)}
    if fmt == "svg":
        from PyQt6.QtSvg import QSvgGenerator

        gen = QSvgGenerator()
        gen.setFileName(str(path))
        gen.setSize(QSize(figure.width, figure.height))
        gen.setViewBox(QRect(0, 0, figure.width, figure.height))
        gen.setResolution(96)
        gen.setTitle(figure.name)
        gen.setDescription("Exported by MINFLUX Viewer")
        painter = QPainter(gen)
    else:
        writer = QPdfWriter(str(path))
        writer.setResolution(96)
        # A page exactly the size of the figure, one logical pixel per 1/96 in.
        writer.setPageSize(QPageSize(QSizeF(figure.width * 72.0 / 96.0,
                                            figure.height * 72.0 / 96.0),
                                     QPageSize.Unit.Point))
        from PyQt6.QtCore import QMarginsF
        from PyQt6.QtGui import QPageLayout
        writer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Point)
        writer.setTitle(figure.name)
        writer.setCreator("MINFLUX Viewer")
        painter = QPainter(writer)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        with _pyqtgraph_export_mode(figure, 1.0):
            paint_figure(painter, figure)
    finally:
        painter.end()
    if not path.exists():
        raise OSError(f"could not write {path}")
    return {"format": fmt, "width": figure.width, "height": figure.height,
            "path": str(path)}


def default_path(figure: Figure, fmt: str, folder: str | None = None) -> Path:
    """The suggested, not yet taken, file for *figure* in *folder*."""
    base = Path(folder or figure.folder_hint or Path.home())
    return next_free_path(base, suggested_stem(figure), FORMATS[fmt][1])


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

def _settings(state) -> dict:
    prefs = getattr(state, "prefs", None)
    if not isinstance(prefs, dict):
        return {}
    return prefs.setdefault("figure_export", {})


class FigureExportDialog:
    """Modal: choose pixel or vector, the format, the scale and the file.

    *memory* is the exporting view's own record of its last export
    (``{"path", "auto_base"}``), updated on success: the next suggestion from the
    same view continues from it.
    """

    def __init__(self, parent, figure: Figure, state=None, memory: dict | None = None) -> None:
        from PyQt6.QtWidgets import (
            QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
            QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QRadioButton,
            QSpinBox, QVBoxLayout)

        from .file_manager import reveal_in_file_manager

        self._figure = figure
        self._state = state
        self._memory = memory if memory is not None else {}
        self._file_dialog = QFileDialog
        self._reveal = reveal_in_file_manager
        self._confirmed: Path | None = None     # an overwrite already confirmed by Browse
        self._final_path: Path | None = None
        settings = _settings(state)
        dlg = self.dialog = QDialog(parent)
        dlg.setWindowTitle("Export Figure")
        root = QVBoxLayout(dlg)
        root.addWidget(QLabel(
            f"<b>{figure.name}</b> &nbsp;—&nbsp; {figure.width} × {figure.height} px on "
            "screen. The view is exported exactly as shown."))

        grid = QGridLayout()
        self._pixel = QRadioButton("Pixel image")
        self._vector = QRadioButton("Vector figure")
        group = QButtonGroup(dlg)
        group.addButton(self._pixel)
        group.addButton(self._vector)
        self._pixel.setToolTip(
            "One image of the view; PNG, TIFF and JPEG carry the same pixels.")
        self._vector.setToolTip(
            "Axes, text, lines and markers stay vector; an image layer (a render,\n"
            "a density map) is embedded as the image it is on screen.")
        grid.addWidget(QLabel("Kind:"), 0, 0)
        kinds = QHBoxLayout()
        kinds.addWidget(self._pixel)
        kinds.addWidget(self._vector)
        kinds.addStretch(1)
        grid.addLayout(kinds, 0, 1)
        grid.addWidget(QLabel("Format:"), 1, 0)
        self._format = QComboBox()
        grid.addWidget(self._format, 1, 1)
        grid.addWidget(QLabel("Scale:"), 2, 0)
        scale_row = QHBoxLayout()
        self._scale = QSpinBox()
        self._scale.setRange(1, MAX_SCALE)
        self._scale.setSuffix(" ×")
        self._scale.setToolTip(
            "Multiple of the screen resolution. Lines, text and markers are drawn\n"
            "again at that size; an image layer is resampled.")
        self._size_label = QLabel("")
        scale_row.addWidget(self._scale)
        scale_row.addWidget(self._size_label)
        scale_row.addStretch(1)
        grid.addLayout(scale_row, 2, 1)
        grid.addWidget(QLabel("File:"), 3, 0)
        path_row = QHBoxLayout()
        self._path = QLineEdit()
        browse = QPushButton("Browse…")
        path_row.addWidget(self._path, 1)
        path_row.addWidget(browse)
        grid.addLayout(path_row, 3, 1)
        self._open_folder = QCheckBox("Open saved folder when completed")
        self._open_folder.setToolTip("Show the exported file in the file manager.")
        grid.addWidget(self._open_folder, 4, 1)
        root.addLayout(grid)
        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet("color: gray; font-size: 11px;")
        root.addWidget(self._note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(dlg.reject)
        root.addWidget(buttons)

        if not figure.vector_ok:
            self._vector.setEnabled(False)
            self._vector.setToolTip(
                "A 3-D view is a rendered image: export it as a pixel image.")
        fmt = settings.get("format", "png")
        vector = fmt in VECTOR_FORMATS and figure.vector_ok
        (self._vector if vector else self._pixel).setChecked(True)
        self._scale.setValue(int(settings.get("scale", 2)))
        self._open_folder.setChecked(bool(settings.get("open_folder", False)))
        self._fill_formats(fmt)
        self._folder = settings.get("folder")
        self._auto_text = str(self._suggest(self.fmt()))
        self._path.setText(self._auto_text)

        self._pixel.toggled.connect(lambda *_: self._on_kind())
        self._format.currentIndexChanged.connect(lambda *_: self._on_format())
        self._scale.valueChanged.connect(lambda *_: self._update_size())
        browse.clicked.connect(self._browse)
        self._update_size()
        self._update_note()

    def fmt(self) -> str:
        return str(self._format.currentData() or "png")

    def _suggest(self, fmt: str) -> Path:
        """The next free file for this view in *fmt*.

        A view that exported before continues from its last file — its folder,
        and its name when the user typed one; otherwise the name is built from
        the dataset, plot type and current view mode.
        """
        ext = FORMATS[fmt][1]
        stem = suggested_stem(self._figure)
        folder = self._folder or self._figure.folder_hint or str(Path.home())
        last = self._memory.get("path")
        if last:
            last = Path(last)
            folder = str(last.parent)
            typed = base_stem(last.stem)
            if typed != self._memory.get("auto_base"):
                stem = typed
        return next_free_path(folder, stem, ext)

    def _fill_formats(self, preferred: str) -> None:
        table = PIXEL_FORMATS if self._pixel.isChecked() else VECTOR_FORMATS
        self._format.blockSignals(True)
        self._format.clear()
        for key, (label, ext) in table.items():
            self._format.addItem(f"{label} ({ext})", key)
        i = self._format.findData(preferred)
        self._format.setCurrentIndex(max(i, 0))
        self._format.blockSignals(False)
        self._scale.setEnabled(self._pixel.isChecked())

    def _on_kind(self) -> None:
        self._fill_formats(self.fmt())
        self._on_format()

    def _on_format(self) -> None:
        ext = FORMATS[self.fmt()][1]
        text = self._path.text().strip()
        if not text or text == self._auto_text:
            # Still the suggestion: suggest again, so the counter is right for
            # the new extension.
            self._auto_text = str(self._suggest(self.fmt()))
            self._path.setText(self._auto_text)
        else:
            self._path.setText(str(with_extension(text, ext)))
        self._update_size()
        self._update_note()

    def _update_size(self) -> None:
        if self._pixel.isChecked():
            s = self._scale.value()
            self._size_label.setText(
                f"→ {self._figure.width * s} × {self._figure.height * s} px")
        else:
            self._size_label.setText("")

    def _update_note(self) -> None:
        notes = []
        if self._figure.is_3d:
            notes.append("A 3-D view is captured from the GPU, as a screenshot would be.")
        elif self._figure.has_opengl and self._vector.isChecked():
            notes.append("Points drawn by the GPU are embedded as an image; axes and "
                         "text stay vector.")
        if self.fmt() == "jpg":
            notes.append("JPEG is lossy; prefer PNG or TIFF for figures.")
        self._note.setText(" ".join(notes))

    def _browse(self) -> None:
        ext = FORMATS[self.fmt()][1]
        label = FORMATS[self.fmt()][0]
        path, _ = self._file_dialog.getSaveFileName(
            self.dialog, "Export Figure", self._path.text(), f"{label} (*{ext})")
        if path:
            chosen = with_extension(path, ext)
            # The file dialog asked about replacing the name it returned; a name
            # this dialog had to change is asked about again on Save.
            self._confirmed = chosen if chosen == Path(path) else None
            self._path.setText(str(chosen))

    def _target(self) -> Path | None:
        text = self._path.text().strip()
        if not text:
            return None
        return with_extension(text, FORMATS[self.fmt()][1])

    def _on_save(self) -> None:
        """Validate before closing: a missing name or folder, and a file that is
        already there (the usual replace question), keep the dialog open."""
        from PyQt6.QtWidgets import QMessageBox

        path = self._target()
        if path is None or not path.name:
            QMessageBox.warning(self.dialog, "Export Figure", "Choose a file to write.")
            return
        if not path.parent.is_dir():
            QMessageBox.warning(self.dialog, "Export Figure",
                                f"The folder {path.parent} does not exist.")
            return
        if path.exists() and path != self._confirmed:
            answer = QMessageBox.question(
                self.dialog, "Confirm Save As",
                f"{path.name} already exists.\nDo you want to replace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._final_path = path
        self.dialog.accept()

    def run(self) -> dict | None:
        from PyQt6.QtWidgets import QDialog, QMessageBox

        if self.dialog.exec() != QDialog.DialogCode.Accepted or self._final_path is None:
            return None
        path = self._final_path
        try:
            result = save_figure(self._figure, path, self.fmt(), scale=self._scale.value())
        except Exception as exc:                            # noqa: BLE001
            QMessageBox.warning(self.dialog, "Export Figure", f"Could not export:\n{exc}")
            return None
        settings = _settings(self._state)
        settings.update(format=self.fmt(), scale=self._scale.value(), folder=str(path.parent),
                        open_folder=self._open_folder.isChecked())
        self._memory.update(path=str(path), auto_base=suggested_stem(self._figure))
        if self._open_folder.isChecked():
            try:
                self._reveal(path)
            except Exception:                              # noqa: BLE001 - never fail the export
                pass
        return result


def export_figure_interactive(parent, figure: Figure, state=None) -> dict | None:
    """Ask, export, and report to the Log.

    The exporting view (*parent*) keeps a small memory of its last export, so the
    next export from it continues the numbering.
    """
    memory = getattr(parent, "_figure_export_memory", None)
    if memory is None:
        memory = {}
        try:
            parent._figure_export_memory = memory
        except AttributeError:
            pass
    result = FigureExportDialog(parent, figure, state, memory=memory).run()
    if result is not None and state is not None and hasattr(state, "log"):
        label = FORMATS[result["format"]][0]
        state.log(f"Exported figure of '{figure.name}' to {result['path']} "
                  f"({label}, {result['width']} × {result['height']} px).")
    return result
