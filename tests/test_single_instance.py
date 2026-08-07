"""One viewer process per user; later launches hand their files over.

Backs the macOS "dropping a .msr opens a second copy" fix without relying on
Launch Services honouring LSMultipleInstancesProhibited. Uses QLocalServer /
QLocalSocket, so these tests exercise the real transport on every platform.
"""

from __future__ import annotations

import pytest

from minflux_viewer.ui.single_instance import (
    ALLOW_MULTIPLE_ENV,
    SingleInstanceGuard,
    allow_multiple_instances,
    decode_payload,
    encode_payload,
    server_name,
)


# ------------------------------------------------------------------ payload
def test_payload_round_trips_paths_with_spaces_and_unicode():
    paths = ["/tmp/a b/acq 01.msr", "/tmp/µ-données/Übung.mat", "C:\\data\\x.zarr"]
    assert decode_payload(encode_payload(paths)) == paths


def test_payload_is_newline_framed():
    """The reader frames on the newline, so it must be there exactly once."""
    blob = encode_payload(["/tmp/x.msr"])
    assert blob.endswith(b"\n")
    assert blob.count(b"\n") == 1


def test_empty_request_is_a_raise_request_not_a_path():
    assert decode_payload(encode_payload([])) == []


@pytest.mark.parametrize("junk", [b"", b"not json\n", b"[1,2,3]\n", b"\xff\xfe\n"])
def test_malformed_payloads_yield_no_paths(junk):
    """A malformed hand-off must not crash the running instance."""
    assert decode_payload(junk) == []


# ------------------------------------------------------------------- opt-out
@pytest.mark.parametrize("value,expected", [
    ("", False), ("0", False), ("false", False), ("no", False),
    ("1", True), ("true", True), ("yes", True),
])
def test_allow_multiple_reads_the_environment(value, expected):
    assert allow_multiple_instances({ALLOW_MULTIPLE_ENV: value}) is expected


def test_allow_multiple_defaults_to_off():
    assert allow_multiple_instances({}) is False


def test_server_name_is_scoped_to_the_user():
    """Two accounts on one machine must not fight over a single viewer."""
    name = server_name()
    assert name.startswith("minflux-viewer-")
    assert server_name() == name           # stable within a process


# -------------------------------------------------------------- hand-off
@pytest.fixture
def guard_name(request):
    return f"minflux-viewer-test-{abs(hash(request.node.nodeid)) % 10**8}"


def test_no_primary_means_this_process_should_continue(qapp, guard_name):
    """With nobody listening, hand_off must report False so we build a UI."""
    guard = SingleInstanceGuard(guard_name)
    assert guard.hand_off_to_primary(["/tmp/x.msr"]) is False


def _hand_off_from_a_real_process(name: str, paths: list[str]):
    """Run the hand-off in a separate process, as a real second launch is.

    It cannot be done in-process: ``hand_off_to_primary`` blocks in
    ``waitForConnected`` / ``waitForDisconnected``, which does not pump the
    primary's event loop, so a same-process 'second launch' would deadlock
    against the server it is trying to reach. A subprocess also exercises the
    only thing that matters here — that two OS processes can rendezvous.
    """
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from PyQt6.QtCore import QCoreApplication\n"
        "app = QCoreApplication([])\n"
        "from minflux_viewer.ui.single_instance import SingleInstanceGuard\n"
        "ok = SingleInstanceGuard(%r).hand_off_to_primary(%r)\n"
        "sys.exit(0 if ok else 3)\n" % (str(root), name, paths)
    )
    return subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_second_launch_hands_its_files_to_the_primary(qapp, qtbot, guard_name):
    primary = SingleInstanceGuard(guard_name)
    assert primary.listen() is True
    received: list[str] = []
    primary.path_received.connect(received.append)

    proc = _hand_off_from_a_real_process(guard_name, ["/tmp/one.msr", "/tmp/two.mat"])
    try:
        qtbot.waitUntil(lambda: len(received) == 2, timeout=15000)
        assert received == ["/tmp/one.msr", "/tmp/two.mat"]
        assert proc.wait(timeout=15) == 0      # it reported the hand-off worked
    finally:
        proc.kill()
        primary.stop()


def test_second_launch_without_files_asks_the_primary_to_come_forward(
        qapp, qtbot, guard_name):
    """Double-clicking the app while it runs must raise it, not do nothing."""
    primary = SingleInstanceGuard(guard_name)
    assert primary.listen() is True
    raised: list[bool] = []
    primary.raise_requested.connect(lambda: raised.append(True))

    proc = _hand_off_from_a_real_process(guard_name, [])
    try:
        qtbot.waitUntil(lambda: raised == [True], timeout=15000)
        assert proc.wait(timeout=15) == 0
    finally:
        proc.kill()
        primary.stop()


def test_a_second_launch_exits_rather_than_opening_a_window(qapp, qtbot, guard_name):
    """The whole point: the duplicate process must terminate, not show a UI.

    This is what stops a dropped .msr from producing a second viewer even when
    Launch Services ignores LSMultipleInstancesProhibited.
    """
    primary = SingleInstanceGuard(guard_name)
    assert primary.listen() is True
    seen: list[str] = []
    primary.path_received.connect(seen.append)

    proc = _hand_off_from_a_real_process(guard_name, ["/tmp/dropped.msr"])
    try:
        qtbot.waitUntil(lambda: seen == ["/tmp/dropped.msr"], timeout=15000)
        assert proc.wait(timeout=15) == 0      # exit 0 == "handed over, stopping"
    finally:
        proc.kill()
        primary.stop()


def test_a_stopped_primary_no_longer_accepts_hand_offs(qapp, guard_name):
    """While shutting down we must refuse, so the newcomer opens the files."""
    primary = SingleInstanceGuard(guard_name)
    assert primary.listen() is True
    primary.stop()

    assert SingleInstanceGuard(guard_name).hand_off_to_primary(["/tmp/x.msr"]) is False


def test_listen_reclaims_a_socket_left_by_a_crashed_run(qapp, guard_name):
    """A crash can leave the socket behind; the next run must still start."""
    stale = SingleInstanceGuard(guard_name)
    assert stale.listen() is True
    stale._server.newConnection.disconnect()
    stale._server = None                   # leak it, as a crash would

    survivor = SingleInstanceGuard(guard_name)
    assert survivor.listen() is True
    survivor.stop()


def test_the_guard_is_disabled_by_the_environment(qapp, guard_name, monkeypatch):
    monkeypatch.setenv(ALLOW_MULTIPLE_ENV, "1")
    guard = SingleInstanceGuard(guard_name)
    assert guard.listen() is False
    assert guard.hand_off_to_primary(["/tmp/x.msr"]) is False


def test_multiple_second_launches_are_all_delivered(qapp, qtbot, guard_name):
    """Dropping several files in a row must not lose any of them."""
    primary = SingleInstanceGuard(guard_name)
    assert primary.listen() is True
    received: list[str] = []
    primary.path_received.connect(received.append)

    procs = [_hand_off_from_a_real_process(guard_name, [f"/tmp/{i}.msr"])
             for i in range(3)]
    try:
        qtbot.waitUntil(lambda: len(received) == 3, timeout=20000)
        assert sorted(received) == ["/tmp/0.msr", "/tmp/1.msr", "/tmp/2.msr"]
    finally:
        for proc in procs:
            proc.kill()
        primary.stop()
