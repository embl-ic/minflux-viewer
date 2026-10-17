"""
minflux_viewer.core.self_update
===============================
One-click self-update for the **frozen (PyInstaller one-folder) Windows** build.

A running one-folder app holds its own ``minflux_viewer.exe`` + ``_internal`` DLLs
**locked**, so the new files can't be copied over the install in place. The only
reliable pattern is to hand off to a **helper that runs after the app exits**:

    1. (dialog) download the release ``-win.zip`` asset  → a temp file
    2. stage_update()      extract it → find the folder containing the exe
    3. build_helper_bat()  a .bat that: waits for our PID to exit → backs up the
       install dir → robocopy the staged files over it → relaunch → rollback on
       failure → self-delete
    4. launch_helper()     start the .bat detached (elevated only if the install
       dir isn't writable), then the app quits

Everything here is Qt-free and stdlib-only so the pure pieces are unit-tested;
only :func:`launch_helper` actually spawns a process / touches the real install.
Non-Windows / non-frozen runs return ``can_self_update() == False`` and the UI
falls back to opening the download page.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import weakref
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

#: Fallback app exe name (real name is taken from the running executable).
APP_EXE_NAME = "minflux_viewer.exe"

# Windows process-creation flags (avoid importing subprocess constants that are
# absent on non-Windows, so this module imports everywhere for testing).
_CREATE_NEW_CONSOLE = 0x00000010
_CREATE_NEW_PROCESS_GROUP = 0x00000200


def is_frozen() -> bool:
    """True when running as a PyInstaller-frozen executable."""
    return bool(getattr(sys, "frozen", False))


def is_windows() -> bool:
    return sys.platform.startswith("win")


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    return sys.platform == "darwin"


def current_exe() -> Path:
    """Path to the running executable (the frozen app exe when frozen)."""
    return Path(sys.executable).resolve()


def install_dir() -> Path:
    """The one-folder install directory (parent of the running exe)."""
    return current_exe().parent


def can_self_update() -> bool:
    """Whether one-click self-update is supported in this run.

    Supported on the **frozen Windows and Linux** builds (they can swap their own
    one-folder install and relaunch). **macOS is intentionally excluded**: a
    downloaded, unsigned ``.app`` is Gatekeeper-quarantined, so an in-place update
    would re-trigger a security prompt every time — a clean macOS auto-update
    needs Developer ID code-signing + notarization first. A source checkout also
    falls back to the download page.
    """
    return is_frozen() and (is_windows() or is_linux())


def find_app_root(extract_dir, exe_name: str = APP_EXE_NAME) -> Path:
    """Locate the folder inside an extracted release zip that holds *exe_name*.

    Handles a zip that wraps the app in a top folder (``minflux_viewer/…``, as
    ``Compress-Archive -Path dist\\minflux_viewer`` produces) as well as one
    that has the exe at the root. Raises ``FileNotFoundError`` if not found.
    """
    root = Path(extract_dir)
    # Match the exe *file* — on Linux the exe (`minflux_viewer`) and its folder
    # share a name, so `.exists()` would wrongly match the directory.
    if (root / exe_name).is_file():
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / exe_name).is_file():
            return child
    for found in root.rglob(exe_name):
        if found.is_file():
            return found.parent
    raise FileNotFoundError(f"{exe_name} not found under {root}")


def is_writable(path) -> bool:
    """Best-effort test that *path* (a directory) is writable without elevation."""
    probe = Path(path) / f".mfv_write_test_{os.getpid()}"
    try:
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def stage_update(archive_path, *, exe_name: str = APP_EXE_NAME) -> tuple[Path, Path]:
    """Extract *archive_path* to a temp dir; return ``(staged_app_root, extract_dir)``.

    Handles both ``.zip`` (Windows asset) and ``.tar.gz``/``.tgz``/``.tar.xz``
    (Linux asset). ``staged_app_root`` is the folder containing *exe_name* (the
    files to copy over the install); ``extract_dir`` is its temp parent (deleted
    by the helper after the swap).
    """
    name = str(archive_path).lower()
    is_tar = name.endswith((".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".tar"))
    if is_tar:
        import tarfile
        # Unfiltered, a member named "../x" or "/x" is written outside the
        # staging folder (on Windows an absolute name lands on the drive root).
        # The "data" filter refuses those, and links or devices that escape;
        # a Python too old to have it is refused rather than trusted.
        if not hasattr(tarfile, "data_filter"):
            raise RuntimeError(
                "This Python cannot extract a .tar archive safely (no tarfile "
                "data filter); download the update from the release page.")
    extract_dir = Path(tempfile.mkdtemp(prefix="mfv_update_"))
    completed = False
    try:
        payload_dir = extract_dir / "payload"
        payload_dir.mkdir()
        if is_tar:
            with tarfile.open(archive_path) as tf:
                tf.extractall(payload_dir, filter="data")
        else:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(payload_dir)
        result = find_app_root(payload_dir, exe_name), extract_dir
        completed = True
        return result
    finally:
        if not completed:
            _remove_staging(extract_dir)


def build_helper_bat(*, pid: int, staged_dir, install_dir, exe_path, extract_dir) -> str:
    """Return the Windows helper batch script text (see module docstring).

    The script waits for *pid* to exit, moves the install to ``.bak``, copies the
    staged files in, relaunches *exe_path*, rolls back on failure, cleans up
    ``extract_dir`` and deletes itself. All paths are embedded (no arg parsing).
    """
    staged = str(Path(staged_dir))
    inst = str(Path(install_dir))
    exe = str(Path(exe_path))
    ext = str(Path(extract_dir))
    return (
        "@echo off\r\n"
        # Never sit in the install dir — Windows can't move a directory that is a
        # running process's current directory (this helper's CWD is inherited).
        'cd /d "%TEMP%"\r\n'
        "echo Updating MINFLUX Viewer. Please wait...\r\n"
        ":waitloop\r\n"
        f'tasklist /fi "PID eq {pid}" 2>nul | find "{pid}" >nul 2>&1\r\n'
        "if not errorlevel 1 (\r\n"
        "    ping -n 2 127.0.0.1 >nul\r\n"
        "    goto waitloop\r\n"
        ")\r\n"
        f'if exist "{inst}.bak" rmdir /s /q "{inst}.bak" >nul 2>&1\r\n'
        f'move "{inst}" "{inst}.bak" >nul 2>&1\r\n'
        "if errorlevel 1 goto failkeep\r\n"
        f'robocopy "{staged}" "{inst}" /e /nfl /ndl /njh /njs /nc /ns /np >nul\r\n'
        "if errorlevel 8 goto rollback\r\n"
        f'rmdir /s /q "{inst}.bak" >nul 2>&1\r\n'
        "goto launch\r\n"
        ":rollback\r\n"
        f'if exist "{inst}" rmdir /s /q "{inst}" >nul 2>&1\r\n'
        f'move "{inst}.bak" "{inst}" >nul 2>&1\r\n'
        ":failkeep\r\n"
        "echo Update could not be applied; your previous version was kept.\r\n"
        ":launch\r\n"
        f'start "" "{exe}"\r\n'
        f'rmdir /s /q "{ext}" >nul 2>&1\r\n'
        'del "%~f0" >nul 2>&1\r\n'
    )


def build_helper_sh(*, pid: int, staged_dir, install_dir, exe_path, extract_dir) -> str:
    """Return the Linux helper shell script (POSIX ``sh``).

    Waits for *pid* to exit, moves the install to ``.bak``, copies the staged
    files in, relaunches *exe_path* (detached), rolls back on failure, cleans up
    ``extract_dir`` and deletes itself. All paths are embedded (no arg parsing).
    """
    # POSIX forward-slash paths (Path.as_posix keeps '/' even when generated on Windows).
    staged = Path(staged_dir).as_posix()
    inst = Path(install_dir).as_posix()
    exe = Path(exe_path).as_posix()
    ext = Path(extract_dir).as_posix()
    return (
        "#!/bin/sh\n"
        f'while kill -0 {pid} 2>/dev/null; do sleep 0.5; done\n'
        f'rm -rf "{inst}.bak"\n'
        f'if mv "{inst}" "{inst}.bak"; then\n'
        f'  if cp -a "{staged}" "{inst}"; then\n'
        f'    rm -rf "{inst}.bak"\n'
        f'  else\n'
        f'    rm -rf "{inst}"; mv "{inst}.bak" "{inst}"\n'
        f'  fi\n'
        f'fi\n'
        f'chmod +x "{exe}" 2>/dev/null\n'
        f'( "{exe}" >/dev/null 2>&1 & )\n'
        f'rm -rf "{ext}"\n'
        'rm -- "$0"\n'
    )


def launch_helper(script_path, target_dir) -> None:
    """Start the swap-and-relaunch helper detached (Windows ``.bat`` / Linux ``.sh``).

    Windows: runs in its own console, elevating via ``ShellExecuteW("runas", …)``
    only if *target_dir* isn't writable (``C:\\Program Files``). Linux: a detached
    ``sh`` session in a new process group so it outlives the exiting app.
    """
    script_path = str(script_path)
    neutral_cwd = tempfile.gettempdir()          # not inside the install dir
    if is_windows():
        if is_writable(target_dir):
            subprocess.Popen(
                ["cmd", "/c", script_path], cwd=neutral_cwd,
                creationflags=_CREATE_NEW_CONSOLE | _CREATE_NEW_PROCESS_GROUP,
                close_fds=True)
        else:
            import ctypes
            # SW_SHOWNORMAL = 1; 'runas' triggers UAC; 5th arg = neutral CWD.
            outcome = ctypes.windll.shell32.ShellExecuteW(  # type: ignore[attr-defined]
                None, "runas", "cmd.exe", f'/c "{script_path}"', neutral_cwd, 1)
            if outcome <= 32:
                raise OSError(f"Could not launch elevated update helper ({outcome}).")
    else:
        subprocess.Popen(
            ["/bin/sh", script_path], cwd=neutral_cwd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True)


@dataclass(frozen=True)
class PreparedUpdate:
    script_path: Path
    extract_dir: Path
    target_dir: Path
    _finalizer: weakref.finalize = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # A queued result can be abandoned between signal emission and UI
        # delivery. Resource ownership must survive that gap without leaking.
        object.__setattr__(self, "_finalizer", weakref.finalize(self, _remove_staging, self.extract_dir))


def _remove_staging(directory: Path) -> None:
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        return
    except OSError:
        logging.getLogger(__name__).exception("Could not remove update staging directory")


def discard_prepared_update(prepared: PreparedUpdate) -> None:
    """Release an unlaunched helper and only its owned staging directory."""
    prepared._finalizer()


def launch_prepared_update(prepared: PreparedUpdate) -> None:
    """Transfer cleanup ownership to the helper only after a successful launch."""
    if not prepared._finalizer.alive:
        raise ValueError("Prepared update was already discarded or handed off.")
    launch_helper(prepared.script_path, prepared.target_dir)
    prepared._finalizer.detach()


def prepare_update(archive_path, *, pid: int | None = None, exe_path=None,
                   target_dir=None) -> PreparedUpdate:
    """Extract already-authenticated bytes and build, but do not launch, a helper.

    This is blocking I/O; the GUI schedules it on the shared file-I/O pool.
    The private staging directory also owns the helper (no predictable temp
    script name). Authentication is the caller's responsibility.
    """
    exe = Path(exe_path) if exe_path else current_exe()
    target = Path(target_dir) if target_dir else install_dir()
    the_pid = int(pid if pid is not None else os.getpid())

    staged_root, extract_dir = stage_update(archive_path, exe_name=exe.name)
    completed = False
    try:
        kw = dict(pid=the_pid, staged_dir=staged_root, install_dir=target,
                  exe_path=exe, extract_dir=extract_dir)
        if is_windows():
            script = build_helper_bat(**kw)
            script_path = extract_dir / "apply-update.bat"
            script_path.write_text(script, encoding="ascii")
        else:
            script = build_helper_sh(**kw)
            script_path = extract_dir / "apply-update.sh"
            script_path.write_text(script, encoding="utf-8")
            os.chmod(script_path, 0o755)
        completed = True
        return PreparedUpdate(script_path, extract_dir, target)
    finally:
        if not completed:
            _remove_staging(extract_dir)


def apply_update(archive_path, *, pid: int | None = None, exe_path=None,
                 target_dir=None) -> Path:
    """Blocking low-level staging/handoff; caller authenticates and then quits.

    The GUI uses signed_updates.prepare_download on an I/O worker instead.
    This compatibility helper does not establish publisher authenticity.
    """
    prepared = prepare_update(archive_path, pid=pid, exe_path=exe_path, target_dir=target_dir)
    launched = False
    try:
        launch_prepared_update(prepared)
        launched = True
        return prepared.script_path
    finally:
        if not launched:
            discard_prepared_update(prepared)
