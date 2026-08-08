"""macOS document relaying without a global single-instance policy."""

from __future__ import annotations

import pytest

from minflux_viewer.ui.document_open_relay import (
    ALLOW_MULTIPLE_ENV,
    NEW_INSTANCE_ARG,
    DocumentOpenRelay,
    decode_reply,
    decode_request,
    decode_request_details,
    encode_reply,
    encode_request,
    server_name,
    should_handoff_documents,
)


# ------------------------------------------------------------------ policy
def test_no_document_launch_is_never_suppressed():
    """An intentional second viewer has no document and must build its own UI."""
    assert not should_handoff_documents([], platform="darwin", argv=[], env={})


def test_only_macos_document_startups_use_the_fallback():
    paths = ["/tmp/acquisition.msr"]
    assert should_handoff_documents(paths, platform="darwin", argv=[], env={})
    assert not should_handoff_documents(paths, platform="win32", argv=[], env={})
    assert not should_handoff_documents(paths, platform="linux", argv=[], env={})


def test_new_instance_argument_keeps_documents_in_the_new_process():
    assert not should_handoff_documents(
        ["/tmp/acquisition.msr"],
        platform="darwin",
        argv=[NEW_INSTANCE_ARG],
        env={},
    )


@pytest.mark.parametrize("value", ["1", "true", "yes"])
def test_legacy_allow_multiple_environment_still_forces_a_new_process(value):
    assert not should_handoff_documents(
        ["/tmp/acquisition.msr"],
        platform="darwin",
        argv=[],
        env={ALLOW_MULTIPLE_ENV: value},
    )


# ---------------------------------------------------------------- payload
def test_request_round_trips_paths_with_spaces_and_unicode():
    paths = ["/tmp/a b/acq 01.msr", "/tmp/µ-données/Übung.mat", "C:\\data\\x.zarr"]
    assert decode_request(encode_request(paths)) == paths
    details = decode_request_details(encode_request(paths))
    assert details is not None
    assert details["paths"] == paths
    assert isinstance(details["pid"], int)
    assert details["executable"]


def test_protocol_messages_are_newline_framed():
    request = encode_request(["/tmp/x.msr"])
    reply = encode_reply(True)
    assert request.endswith(b"\n") and request.count(b"\n") == 1
    assert reply.endswith(b"\n") and reply.count(b"\n") == 1
    assert decode_reply(reply) is True
    assert decode_reply(encode_reply(False)) is False


@pytest.mark.parametrize("junk", [b"", b"not json\n", b"[1,2,3]\n", b"\xff\xfe\n"])
def test_malformed_requests_are_rejected(junk):
    assert decode_request(junk) == []


def test_server_name_is_scoped_to_the_user_and_purpose():
    name = server_name()
    assert name.startswith("minflux-viewer-document-relay-")
    assert server_name() == name


# -------------------------------------------------------------- transport
@pytest.fixture
def relay_name(request):
    return f"minflux-viewer-document-test-{abs(hash(request.node.nodeid)) % 10**8}"


def _handoff_from_a_real_process(name: str, paths: list[str]):
    """Use a process because the blocking client cannot share the broker loop."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        f"import sys; sys.path.insert(0, {str(root)!r})\n"
        "from PyQt6.QtCore import QCoreApplication\n"
        "app = QCoreApplication([])\n"
        "from minflux_viewer.ui.document_open_relay import DocumentOpenRelay\n"
        f"ok = DocumentOpenRelay({name!r}).hand_off_documents({paths!r})\n"
        "sys.exit(0 if ok else 3)\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_no_broker_means_document_process_continues(qapp, relay_name):
    relay = DocumentOpenRelay(relay_name)
    assert relay.hand_off_documents(["/tmp/x.msr"]) is False


def test_empty_request_never_hands_off(qapp, relay_name):
    broker = DocumentOpenRelay(relay_name)
    assert broker.start() is True
    try:
        assert DocumentOpenRelay(relay_name).hand_off_documents([]) is False
    finally:
        broker.stop()


def test_document_launch_is_acknowledged_and_delivered(qapp, qtbot, relay_name):
    broker = DocumentOpenRelay(relay_name)
    assert broker.start() is True
    received: list[str] = []
    requests: list[dict] = []
    broker.path_received.connect(received.append)
    broker.request_received.connect(requests.append)

    proc = _handoff_from_a_real_process(relay_name, ["/tmp/one.msr", "/tmp/two.mat"])
    try:
        qtbot.waitUntil(lambda: received == ["/tmp/one.msr", "/tmp/two.mat"],
                        timeout=15000)
        assert proc.wait(timeout=15) == 0
        assert len(requests) == 1
        assert requests[0]["paths"] == ["/tmp/one.msr", "/tmp/two.mat"]
        assert isinstance(requests[0]["pid"], int)
    finally:
        proc.kill()
        broker.stop()


def test_shutdown_rejects_instead_of_falsely_acknowledging(qapp, qtbot, relay_name):
    broker = DocumentOpenRelay(relay_name)
    assert broker.start() is True
    received: list[str] = []
    broker.path_received.connect(received.append)
    broker.begin_shutdown()

    proc = _handoff_from_a_real_process(relay_name, ["/tmp/late.msr"])
    try:
        qtbot.waitUntil(lambda: proc.poll() is not None, timeout=15000)
        assert proc.returncode == 3          # rejected: caller must continue
        assert received == []
    finally:
        proc.kill()
        broker.stop()


def test_stopped_broker_no_longer_accepts_documents(qapp, relay_name):
    broker = DocumentOpenRelay(relay_name)
    assert broker.start() is True
    broker.stop()
    assert DocumentOpenRelay(relay_name).hand_off_documents(["/tmp/x.msr"]) is False


def test_independent_viewer_can_take_over_after_broker_exits(qapp, qtbot, relay_name):
    first = DocumentOpenRelay(relay_name)
    second = DocumentOpenRelay(relay_name)
    assert first.start() is True
    assert second.start() is False           # it still represents a live viewer

    first.stop()
    try:
        qtbot.waitUntil(
            lambda: second._server is not None and second._server.isListening(),
            timeout=6000,
        )
    finally:
        second.stop()


def test_multiple_document_launches_are_all_delivered(qapp, qtbot, relay_name):
    broker = DocumentOpenRelay(relay_name)
    assert broker.start() is True
    received: list[str] = []
    broker.path_received.connect(received.append)
    procs = [
        _handoff_from_a_real_process(relay_name, [f"/tmp/{index}.msr"])
        for index in range(3)
    ]
    try:
        qtbot.waitUntil(lambda: len(received) == 3, timeout=20000)
        assert sorted(received) == ["/tmp/0.msr", "/tmp/1.msr", "/tmp/2.msr"]
        assert all(proc.wait(timeout=15) == 0 for proc in procs)
    finally:
        for proc in procs:
            proc.kill()
        broker.stop()
