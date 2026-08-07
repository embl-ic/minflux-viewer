#!/usr/bin/env python3
"""Check a built macOS ``.app`` for the things that cause a second instance.

Run on the Mac, against the bundle you actually launched::

    python3 scripts/check_macos_bundle.py "dist/MINFLUX Viewer.app"

It answers, in order, the questions that matter when a dropped file opens a
second copy of the viewer:

1. Does this build contain the fix at all, or was it built from an older tree?
2. Did the Info.plist keys make it into the bundle?
3. Are there other copies of the same bundle identifier registered, so Launch
   Services can hand the document to a different copy than the running one?

Read-only; it changes nothing.
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

#: Modules that must be inside the bundle for the fix to be present.
REQUIRED_MODULES = ("single_instance", "file_open_app")


def _find_plist(app: Path) -> Path:
    plist = app / "Contents" / "Info.plist"
    if not plist.is_file():
        raise SystemExit(f"not an app bundle (no Contents/Info.plist): {app}")
    return plist


def check_code_present(app: Path) -> bool:
    """Whether the bundle carries the single-instance / open-document code.

    PyInstaller packs pure Python into ``base_library.zip`` + a PYZ inside the
    executable, so grep the whole bundle rather than looking for .py files.
    """
    print("1. Is the fix in this build?")
    hits = {}
    for name in REQUIRED_MODULES:
        needle = name.encode()
        found = False
        for path in app.rglob("*"):
            if not path.is_file():
                continue
            try:
                if needle in path.read_bytes():
                    found = True
                    break
            except (OSError, MemoryError):
                continue
        hits[name] = found
        print(f"   {'OK  ' if found else 'MISS'} {name}")
    if not all(hits.values()):
        print("   -> This bundle predates the fix. Rebuild from a tree that has it:")
        print("      git log --oneline -1   # expect the macOS open-document commit")
        return False
    return True


def check_plist(app: Path) -> bool:
    print("\n2. Did the Info.plist keys land?")
    with _find_plist(app).open("rb") as handle:
        plist = plistlib.load(handle)

    ok = True
    types = plist.get("CFBundleDocumentTypes") or []
    if not types:
        print("   MISS CFBundleDocumentTypes  <- Launch Services has no handler,")
        print("        so it launches a new copy instead of using the running one")
        ok = False
    else:
        exts = sorted({e.lower()
                       for entry in types
                       for e in entry.get("CFBundleTypeExtensions", [])})
        print(f"   OK   CFBundleDocumentTypes: {', '.join(exts)}")
        for entry in types:
            rank = entry.get("LSHandlerRank")
            if rank != "Alternate":
                print(f"   note LSHandlerRank={rank!r} for "
                      f"{entry.get('CFBundleTypeName')!r} (expected 'Alternate')")

    multi = plist.get("LSMultipleInstancesProhibited")
    print(f"   {'OK  ' if multi else 'MISS'} LSMultipleInstancesProhibited: {multi!r}")
    ok = ok and bool(multi)

    print(f"   ..   CFBundleIdentifier: {plist.get('CFBundleIdentifier')!r}")
    print(f"   ..   CFBundleShortVersionString: "
          f"{plist.get('CFBundleShortVersionString')!r}")
    return ok


def check_other_copies(app: Path) -> None:
    """Other bundles with the same identifier confuse Launch Services.

    A document can be handed to a copy that is not the one you are running,
    which is indistinguishable from 'it opened a second instance'.
    """
    print("\n3. Other copies of this app on this Mac")
    with _find_plist(app).open("rb") as handle:
        bundle_id = plistlib.load(handle).get("CFBundleIdentifier")
    if not bundle_id:
        print("   (no CFBundleIdentifier; skipping)")
        return
    try:
        out = subprocess.run(
            ["mdfind", f"kMDItemCFBundleIdentifier == '{bundle_id}'"],
            capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"   (mdfind unavailable: {exc})")
        return
    copies = [line for line in out.splitlines() if line.strip()]
    here = str(app.resolve())
    if len(copies) <= 1:
        print(f"   OK   only this one: {here}")
        return
    print(f"   note {len(copies)} copies share id {bundle_id!r}:")
    for copy in copies:
        print(f"          {'* ' if copy == here else '  '}{copy}")
    print("   -> Launch Services may hand a dropped file to a copy other than the")
    print("      one you are running. Keep ONE, then refresh the database:")
    print("      /System/Library/Frameworks/CoreServices.framework/Frameworks"
          "/LaunchServices.framework/Support/lsregister -kill -r -domain local"
          " -domain system -domain user")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    app = Path(argv[1]).expanduser()
    if not app.exists():
        raise SystemExit(f"no such path: {app}")
    if sys.platform != "darwin":
        print("note: not running on macOS; step 3 needs mdfind and will be skipped.\n")

    code_ok = check_code_present(app)
    plist_ok = check_plist(app)
    if sys.platform == "darwin":
        check_other_copies(app)

    print("\nSummary")
    if code_ok and plist_ok:
        print("  Build looks correct. If a drop still opens a second window, the")
        print("  single-instance guard will now catch it: check the Log window for")
        print("  'Open request (..., pid ...)' / 'second launch, handed over'.")
    else:
        print("  Rebuild (and/or refresh Launch Services) before testing again.")
    return 0 if (code_ok and plist_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
