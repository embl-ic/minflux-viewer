"""Keep one viewer process per user, and hand new files to the running one.

Why this exists in addition to the macOS plist keys
---------------------------------------------------
``CFBundleDocumentTypes`` + ``LSMultipleInstancesProhibited`` (see
``minflux_viewer.spec``) ask *Launch Services* to route a dropped document to
the running instance and to refuse a second copy. That is the right thing to
declare, but it is not reliable in practice:

* Launch Services caches bundle metadata, so a copy that was registered before
  the keys existed keeps its old registration until the database is refreshed;
* several copies of the same bundle identifier on disk (``dist/``, an extracted
  release zip, ``/Applications``) are separate registrations, and a document can
  be handed to a *different copy* than the one already running — which looks
  exactly like a second instance;
* an unsigned, quarantined, or ad-hoc-run bundle is not always treated as a
  registered application at all.

None of that is under our control, and none of it is testable from the build
machine. This guard is, and it does not depend on any of it: the first process
listens on a per-user local socket, and any later process hands over its file
arguments and exits immediately. It works the same on Windows (named pipes),
macOS and Linux (Unix domain sockets).

Set ``MINFLUX_VIEWER_ALLOW_MULTIPLE=1`` to opt out and run several copies.
"""

from __future__ import annotations

import json
import os

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

#: Env var that disables the guard entirely.
ALLOW_MULTIPLE_ENV = "MINFLUX_VIEWER_ALLOW_MULTIPLE"

#: How long a starting process waits for the primary to accept the hand-off.
_CONNECT_TIMEOUT_MS = 400
_WRITE_TIMEOUT_MS = 1000


def allow_multiple_instances(env=None) -> bool:
    """Whether the single-instance guard is disabled by the environment."""
    value = (env if env is not None else os.environ).get(ALLOW_MULTIPLE_ENV, "")
    return str(value).strip().lower() not in ("", "0", "false", "no")


def server_name() -> str:
    """Per-user socket name.

    Scoped to the user so two accounts on one machine (fast user switching, a
    terminal server) each get their own viewer rather than fighting over one.
    """
    try:
        user = os.getlogin()
    except OSError:                       # no controlling terminal (services, CI)
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "default"
    uid = getattr(os, "getuid", lambda: "")()
    return f"minflux-viewer-{user}-{uid}".replace(os.sep, "-")


def encode_payload(paths) -> bytes:
    """Serialize an open request. Newline-terminated so the reader can frame it."""
    return (json.dumps({"paths": [str(p) for p in paths]}) + "\n").encode("utf-8")


def decode_payload(data: bytes) -> list[str]:
    """Paths from one encoded request; malformed input yields no paths."""
    try:
        obj = json.loads(bytes(data).decode("utf-8").strip() or "{}")
    except (ValueError, UnicodeDecodeError):
        return []
    if not isinstance(obj, dict):
        return []
    return [str(p) for p in obj.get("paths", []) if str(p)]


class SingleInstanceGuard(QObject):
    """First process listens; later ones hand over their paths and exit."""

    #: Emitted in the primary process for each path a later process sent.
    path_received = pyqtSignal(str)
    #: Emitted when a later process started with no paths — just raise the UI.
    raise_requested = pyqtSignal()

    def __init__(self, name: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self._name = name or server_name()
        self._server: QLocalServer | None = None
        self._buffers: dict = {}

    @property
    def name(self) -> str:
        return self._name

    # -- starting process --------------------------------------------------
    def hand_off_to_primary(self, paths) -> bool:
        """Try to give *paths* to an already-running instance.

        Returns True when a primary accepted them, meaning **this process should
        exit without building a UI**. Returns False when no primary is running
        (or the guard is disabled), meaning this process should continue and
        call :meth:`listen`.
        """
        if allow_multiple_instances():
            return False
        socket = QLocalSocket()
        socket.connectToServer(self._name)
        if not socket.waitForConnected(_CONNECT_TIMEOUT_MS):
            return False
        try:
            socket.write(encode_payload(paths))
            socket.flush()
            socket.waitForBytesWritten(_WRITE_TIMEOUT_MS)
            # Wait for the PRIMARY to close the connection, which it does only
            # after it has read the request (see _consume). Disconnecting from
            # this side instead can tear the pipe down before the peer has
            # accepted it, and the bytes are then dropped rather than queued.
            # Timing out is not fatal — the write already went out — so the
            # hand-off still counts as done and this process still exits.
            socket.waitForDisconnected(_WRITE_TIMEOUT_MS)
        except RuntimeError:
            return False
        finally:
            try:
                socket.abort()
            except RuntimeError:
                pass
        return True

    # -- primary process ---------------------------------------------------
    def listen(self) -> bool:
        """Become the primary. Returns False if the socket could not be claimed.

        A crashed previous run can leave a stale socket behind, which makes
        ``listen`` fail even though nobody is there; ``removeServer`` clears it.
        Only reached after :meth:`hand_off_to_primary` found no live primary, so
        removing it cannot disconnect a running instance.
        """
        if allow_multiple_instances():
            return False
        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        if not server.listen(self._name):
            QLocalServer.removeServer(self._name)
            if not server.listen(self._name):
                return False
        server.newConnection.connect(self._on_new_connection)
        self._server = server
        return True

    def stop(self) -> None:
        """Stop accepting hand-offs.

        Called as the app quits: a process that connects while this one is
        tearing down would otherwise hand over paths that are never opened.
        Better to refuse, so the newcomer becomes the primary and opens them.
        """
        if self._server is not None:
            try:
                self._server.close()
            except RuntimeError:
                pass
            self._server = None

    # -- internals ---------------------------------------------------------
    def _on_new_connection(self) -> None:
        if self._server is None:
            return
        while self._server is not None and self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            if socket is None:
                return
            self._buffers[socket] = bytearray()
            socket.readyRead.connect(lambda s=socket: self._drain(s))
            socket.disconnected.connect(lambda s=socket: self._on_disconnected(s))
            # The sender may already have written and closed before we got here,
            # in which case no readyRead is coming — take whatever is buffered.
            self._drain(socket)

    def _drain(self, socket) -> None:
        """Read whatever is available; consume once a full request has arrived."""
        buffer = self._buffers.get(socket)
        if buffer is None:
            return
        try:
            buffer += bytes(socket.readAll())
        except RuntimeError:
            return
        # One request per connection, framed by the trailing newline.
        if b"\n" in buffer:
            self._consume(socket)

    def _on_disconnected(self, socket) -> None:
        # A sender that closed without a trailing newline still gets handled,
        # and bytes can still be pending on the socket at disconnect time.
        self._drain(socket)
        if self._buffers.get(socket):
            self._consume(socket)
        self._buffers.pop(socket, None)
        try:
            socket.deleteLater()
        except RuntimeError:
            pass

    def _consume(self, socket) -> None:
        data = bytes(self._buffers.pop(socket, b""))
        # Closing from this side is the acknowledgement the sender waits on, so
        # do it whether or not the request parsed — otherwise a malformed
        # hand-off leaves the other process blocked until its timeout.
        try:
            socket.disconnectFromServer()
        except RuntimeError:
            pass
        if not data:
            return
        paths = decode_payload(data)
        if paths:
            for path in paths:
                self.path_received.emit(path)
        else:
            self.raise_requested.emit()
