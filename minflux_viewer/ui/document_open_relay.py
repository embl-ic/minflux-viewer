"""Relay unexpected macOS document launches to an existing viewer.

Normally Launch Services activates a running application and sends it an
Open-Document (``odoc``) Apple Event, which :mod:`file_open_app` receives as a
``QFileOpenEvent``.  Some development/install states can nevertheless start a
second bundle process for the document (for example, when several copies of
the same bundle identifier are registered).

This module is deliberately *not* a single-instance guard.  A process started
without documents always continues and may open another independent viewer.
Only a newly-started macOS process carrying document paths offers those paths
to an existing viewer.  The sender exits only after the receiver explicitly
acknowledges that it accepted the request.
"""

from __future__ import annotations

import json
import os
import sys

from PyQt6.QtCore import QDir, QLockFile, QObject, QTimer, pyqtSignal
from PyQt6.QtNetwork import QLocalServer, QLocalSocket

NEW_INSTANCE_ARG = "--new-instance"

# Backward-compatible escape hatch from the former global single-instance
# guard.  It now means "do not relay startup documents"; no-document launches
# are always allowed regardless of this value.
ALLOW_MULTIPLE_ENV = "MINFLUX_VIEWER_ALLOW_MULTIPLE"

_CONNECT_TIMEOUT_MS = 400
_IO_TIMEOUT_MS = 1000
_RETRY_LISTEN_MS = 1500


def _truthy(value) -> bool:
    return str(value or "").strip().lower() not in ("", "0", "false", "no")


def should_handoff_documents(
    paths,
    *,
    platform: str | None = None,
    argv=None,
    env=None,
) -> bool:
    """Whether this startup should offer its documents to a running viewer.

    The fallback is macOS-only and document-only.  In particular, an ordinary
    second launch with no paths is never handed off or suppressed.  Advanced
    users can also force a separate document-bearing instance with
    ``--new-instance`` or the legacy ``MINFLUX_VIEWER_ALLOW_MULTIPLE=1``.
    """
    current_platform = sys.platform if platform is None else platform
    args = list(sys.argv[1:] if argv is None else argv)
    environment = os.environ if env is None else env
    return bool(
        current_platform == "darwin"
        and list(paths)
        and NEW_INSTANCE_ARG not in args
        and not _truthy(environment.get(ALLOW_MULTIPLE_ENV, ""))
    )


def server_name() -> str:
    """Stable per-user name shared by registered copies of the macOS bundle."""
    try:
        user = os.getlogin()
    except OSError:  # no controlling terminal (app bundles, services, CI)
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "default"
    uid = getattr(os, "getuid", lambda: "")()
    return f"minflux-viewer-document-relay-{user}-{uid}".replace(os.sep, "-")


def encode_request(paths) -> bytes:
    """Serialize one document-open request, including launch diagnostics."""
    payload = {
        "version": 1,
        "kind": "open-documents",
        "paths": [str(path) for path in paths if str(path)],
        "pid": os.getpid(),
        "executable": sys.executable,
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def decode_request(data: bytes) -> list[str]:
    """Return request paths; malformed or non-document payloads are rejected."""
    payload = decode_request_details(data)
    return [] if payload is None else list(payload["paths"])


def decode_request_details(data: bytes) -> dict | None:
    """Validated request details used for diagnostics and path delivery."""
    try:
        payload = json.loads(bytes(data).decode("utf-8").strip() or "{}")
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("kind") != "open-documents":
        return None
    paths = payload.get("paths", [])
    if not isinstance(paths, list):
        return None
    documents = [str(path) for path in paths if str(path)]
    if not documents:
        return None
    pid = payload.get("pid")
    return {
        "paths": documents,
        "pid": int(pid) if isinstance(pid, int) else None,
        "executable": str(payload.get("executable") or ""),
    }


def encode_reply(accepted: bool) -> bytes:
    return (json.dumps({"accepted": bool(accepted)}) + "\n").encode("utf-8")


def decode_reply(data: bytes) -> bool:
    try:
        payload = json.loads(bytes(data).decode("utf-8").strip() or "{}")
    except (ValueError, UnicodeDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("accepted") is True


class DocumentOpenRelay(QObject):
    """Broker document paths without preventing independent viewer processes."""

    path_received = pyqtSignal(str)
    request_received = pyqtSignal(object)

    def __init__(self, name: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self._name = name or server_name()
        self._server: QLocalServer | None = None
        self._lock: QLockFile | None = None
        self._buffers: dict = {}
        self._accepting = True
        self._retry_timer: QTimer | None = None

    @property
    def name(self) -> str:
        return self._name

    def hand_off_documents(self, paths) -> bool:
        """Offer *paths* to a running viewer and wait for explicit acceptance."""
        documents = [str(path) for path in paths if str(path)]
        if not documents:
            return False

        socket = QLocalSocket()
        socket.connectToServer(self._name)
        if not socket.waitForConnected(_CONNECT_TIMEOUT_MS):
            return False
        try:
            if socket.write(encode_request(documents)) < 0:
                return False
            socket.flush()
            if socket.bytesToWrite() and not socket.waitForBytesWritten(_IO_TIMEOUT_MS):
                return False
            if not socket.waitForReadyRead(_IO_TIMEOUT_MS):
                return False
            reply = bytearray(socket.readAll())
            while b"\n" not in reply and socket.waitForReadyRead(_IO_TIMEOUT_MS):
                reply += bytes(socket.readAll())
            return decode_reply(bytes(reply))
        except RuntimeError:
            return False
        finally:
            try:
                socket.abort()
            except RuntimeError:
                pass

    def start(self) -> bool:
        """Try to become broker; followers periodically take over if it exits."""
        self._accepting = True
        if self._listen():
            return True
        if self._retry_timer is None:
            timer = QTimer(self)
            timer.setInterval(_RETRY_LISTEN_MS)
            timer.timeout.connect(self._retry_listen)
            self._retry_timer = timer
        self._retry_timer.start()
        return False

    def begin_shutdown(self) -> None:
        """Keep the name claimed but reject requests during Qt teardown."""
        self._accepting = False
        if self._retry_timer is not None:
            self._retry_timer.stop()

    def stop(self) -> None:
        """Release the relay after the application event loop has stopped."""
        self.begin_shutdown()
        for socket in list(self._buffers):
            try:
                socket.abort()
                socket.deleteLater()
            except RuntimeError:
                pass
        self._buffers.clear()
        if self._server is not None:
            try:
                self._server.close()
            except RuntimeError:
                pass
            self._server = None
        if self._lock is not None:
            try:
                self._lock.unlock()
            except RuntimeError:
                pass
            self._lock = None

    def _retry_listen(self) -> None:
        if self._accepting and self._listen() and self._retry_timer is not None:
            self._retry_timer.stop()

    def _listen(self) -> bool:
        if self._server is not None:
            try:
                if self._server.isListening():
                    return True
            except RuntimeError:
                self._server = None

        # A QLockFile is the election authority. Unlike probing the socket, it
        # records the owner PID and safely recognizes crash residue; a follower
        # can never unlink a live broker's local-server name.
        lock_path = QDir(QDir.tempPath()).filePath(f"{self._name}.lock")
        lock = QLockFile(lock_path)
        lock.setStaleLockTime(0)  # rely on PID/host/process-name checks
        if not lock.tryLock(0):
            return False

        server = QLocalServer(self)
        server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        if not server.listen(self._name):
            # Owning the election lock proves there is no live broker, so this
            # can only be a stale socket left by an abnormal termination.
            QLocalServer.removeServer(self._name)
            if not server.listen(self._name):
                server.deleteLater()
                lock.unlock()
                return False
        server.newConnection.connect(self._on_new_connection)
        self._lock = lock
        self._server = server
        return True

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
            self._drain(socket)

    def _drain(self, socket) -> None:
        buffer = self._buffers.get(socket)
        if buffer is None:
            return
        try:
            buffer += bytes(socket.readAll())
        except RuntimeError:
            return
        if b"\n" in buffer:
            self._consume(socket)

    def _on_disconnected(self, socket) -> None:
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
        request = decode_request_details(data)
        paths = [] if request is None else request["paths"]
        accepted = bool(self._accepting and paths)
        try:
            socket.write(encode_reply(accepted))
            socket.flush()
            if socket.bytesToWrite():
                socket.waitForBytesWritten(_IO_TIMEOUT_MS)
            socket.disconnectFromServer()
        except RuntimeError:
            accepted = False
        if accepted:
            self.request_received.emit(request)
            for path in paths:
                self.path_received.emit(path)
