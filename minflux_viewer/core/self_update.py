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

import os
import subprocess
import sys
import tempfile
import zipfile
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


def current_exe() -> Path:
    """Path to the running executable (the frozen app exe when frozen)."""
    return Path(sys.executable).resolve()


def install_dir() -> Path:
    """The one-folder install directory (parent of the running exe)."""
    return current_exe().parent


def can_self_update() -> bool:
    """Whether one-click self-update is supported in this run.

    Only the **frozen Windows** build can swap its own folder; a source checkout
    or a non-Windows frozen build falls back to the download page.
    """
    return is_frozen() and is_windows()


def find_app_root(extract_dir, exe_name: str = APP_EXE_NAME) -> Path:
    """Locate the folder inside an extracted release zip that holds *exe_name*.

    Handles a zip that wraps the app in a top folder (``minflux_viewer/…``, as
    ``Compress-Archive -Path dist\\minflux_viewer`` produces) as well as one
    that has the exe at the root. Raises ``FileNotFoundError`` if not found.
    """
    root = Path(extract_dir)
    if (root / exe_name).exists():
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / exe_name).exists():
            return child
    for found in root.rglob(exe_name):
        return found.parent
    raise FileNotFoundError(f"{exe_name} not found under {root}")


def is_writable(path) -> bool:
    """Best-effort test that *path* (a directory) is writable without elevation."""
    probe = Path(path) / f".mfv_write_test_{os.getpid()}"
    try:
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except Exception:
        return False


def stage_update(zip_path, *, exe_name: str = APP_EXE_NAME) -> tuple[Path, Path]:
    """Extract *zip_path* to a temp dir; return ``(staged_app_root, extract_dir)``.

    ``staged_app_root`` is the folder containing *exe_name* (the files to copy
    over the install); ``extract_dir`` is its temp parent (deleted by the helper
    after the swap).
    """
    extract_dir = Path(tempfile.mkdtemp(prefix="mfv_update_"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return find_app_root(extract_dir, exe_name), extract_dir


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


def launch_helper(bat_path, target_dir) -> None:
    """Start the helper .bat detached, elevating only if *target_dir* isn't writable.

    When the install dir is user-writable (portable / ``%LOCALAPPDATA%``) the
    helper runs silently in its own console; under ``C:\\Program Files`` it needs
    admin, so it's launched via ``ShellExecuteW("runas", …)`` (a UAC prompt).
    """
    bat_path = str(bat_path)
    neutral_cwd = tempfile.gettempdir()          # not inside the install dir
    if is_writable(target_dir):
        subprocess.Popen(
            ["cmd", "/c", bat_path],
            cwd=neutral_cwd,
            creationflags=_CREATE_NEW_CONSOLE | _CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    else:
        import ctypes

        # SW_SHOWNORMAL = 1; 'runas' triggers the UAC elevation prompt; the 5th
        # arg (lpDirectory) sets the elevated process's CWD to a neutral folder.
        ctypes.windll.shell32.ShellExecuteW(  # type: ignore[attr-defined]
            None, "runas", "cmd.exe", f'/c "{bat_path}"', neutral_cwd, 1)


def apply_update(zip_path, *, pid: int | None = None, exe_path=None,
                 target_dir=None) -> Path:
    """Stage *zip_path* and launch the swap-and-relaunch helper.

    The caller must **quit the app immediately after** this returns so the helper
    (which waits on *pid*) can replace the locked files. Returns the helper .bat
    path. ``pid``/``exe_path``/``target_dir`` default to the running process.
    """
    exe = Path(exe_path) if exe_path else current_exe()
    target = Path(target_dir) if target_dir else install_dir()
    the_pid = int(pid if pid is not None else os.getpid())

    staged_root, extract_dir = stage_update(zip_path, exe_name=exe.name)
    bat = build_helper_bat(pid=the_pid, staged_dir=staged_root, install_dir=target,
                           exe_path=exe, extract_dir=extract_dir)
    bat_path = Path(tempfile.gettempdir()) / f"mfv_apply_update_{the_pid}.bat"
    bat_path.write_text(bat, encoding="ascii")
    launch_helper(bat_path, target)
    return bat_path
