"""A ``QApplication`` that receives the OS "open this document" request.

On **macOS** a file dropped on the application icon (or double-clicked in
Finder, or passed to ``open -a``) is *not* delivered in ``sys.argv``. Launch
Services sends the running process an ``kAEOpenDocuments`` ("odoc") Apple
Event, which Qt translates into a :class:`QFileOpenEvent` on the application
object. An app that does not handle that event silently ignores the file — and
because macOS then has no running-instance handler for the document, it can
launch a *second* copy of the app instead, which is how one dropped ``.msr``
ended up opening a new viewer window.

Two things are needed to make the running instance handle it, and neither works
alone:

* the bundle must declare the document types it opens (``CFBundleDocumentTypes``
  in ``minflux_viewer.spec``), so Launch Services routes the document at all;
* the application must handle :class:`QFileOpenEvent`, which is this module.

Windows and Linux pass file arguments in ``argv`` and never send this event, so
this class is inert there — it is installed unconditionally to keep one code
path on every platform.

**Ordering matters.** The event can arrive *before* the main window exists (it
is delivered as soon as the ``QApplication`` is constructed, and on a cold
launch-with-document that is well before the UI is built), so paths received
before :meth:`set_open_handler` are queued and flushed when the handler
arrives, in order.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication


class FileOpenApplication(QApplication):
    """``QApplication`` that forwards OS open-document requests to a handler."""

    def __init__(self, argv) -> None:
        super().__init__(argv)
        self._pending_paths: list[str] = []
        self._open_handler = None

    # -- Qt ----------------------------------------------------------------
    def event(self, event) -> bool:                        # noqa: D102 - Qt API
        if event.type() == QEvent.Type.FileOpen:
            path = self._path_from_event(event)
            if path:
                self.open_path(path)
                return True
        return super().event(event)

    @staticmethod
    def _path_from_event(event) -> str:
        """Local path carried by a ``QFileOpenEvent``.

        ``file()`` is set for a local file; a ``file://`` URL arrives in
        ``url()`` instead, and a non-file URL yields an empty local path, which
        the caller skips.
        """
        path = ""
        try:
            path = event.file() or ""
            if not path:
                url = event.url()
                path = url.toLocalFile() if url is not None else ""
        except (AttributeError, RuntimeError):
            return ""
        return str(path or "")

    # -- wiring ------------------------------------------------------------
    def set_open_handler(self, handler) -> None:
        """Install *handler* and flush anything that arrived before it.

        Called once the main window exists. ``handler(path)`` is invoked for
        each queued path in arrival order, then for every later request.
        """
        self._open_handler = handler
        pending, self._pending_paths = self._pending_paths, []
        for path in pending:
            self._dispatch(path)

    def open_path(self, path: str) -> None:
        """Handle one open request now, or queue it until a handler exists."""
        if self._open_handler is None:
            self._pending_paths.append(path)
            return
        self._dispatch(path)

    def _dispatch(self, path: str) -> None:
        # A failure here must not kill the event loop: an unhandled exception
        # raised out of QApplication.event() crosses the C++ boundary.
        try:
            self._open_handler(path)
        except Exception:                                  # noqa: BLE001
            import traceback
            traceback.print_exc()

    @property
    def pending_paths(self) -> list[str]:
        """Paths received before a handler was installed (diagnostics/tests)."""
        return list(self._pending_paths)

    def take_pending_paths(self) -> list[str]:
        """Remove and return everything queued so far.

        Used at startup by the single-instance guard: a duplicate launch must
        hand these over to the running instance rather than leave them queued
        for a UI it is about to not build.
        """
        pending, self._pending_paths = self._pending_paths, []
        return pending
