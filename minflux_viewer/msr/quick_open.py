"""Opening a ``.msr`` this application wrote, without the MSR reader.

A vendor ``.msr`` needs the reader: which datasets to import, how to align the
channels, whether to map a confocal image are all questions only the user can
answer. A ``.msr`` **we** wrote has those answers in it already -- the overlay
grouping, the LUTs, the transform, the filters and the ROIs were saved with the
data -- so the reader would ask about things that are settled and then be
overridden by the restored state anyway.

The reader stays reachable from the Plugins menu for these files, exactly as for
any other ``.msr``; this is only about what a plain *open* does.
"""

from __future__ import annotations

import mmap
from pathlib import Path

from .viewer_state import STATE_FORMAT

__all__ = ["is_viewer_msr", "load_viewer_msr"]

#: The marker as it appears in a channel's ``viewer/.zattrs``. That attrs file
#: is plain JSON text inside the MFXDTA archive, which is not itself compressed,
#: so a byte scan finds it -- the same trick ``mfxdta.extract_did_label_map``
#: uses to recover channel labels without decoding any array.
_MARKER = STATE_FORMAT.encode("ascii")


def is_viewer_msr(path) -> bool:
    """True when *path* carries processing state this application wrote.

    A bounded, allocation-free scan: opening a multi-GB acquisition must not
    pay for a full parse merely to decide which route to take.
    """
    source = Path(path)
    try:
        if source.stat().st_size < len(_MARKER):
            return False
        with open(source, "rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
                return data.find(_MARKER) != -1
    except Exception:                                       # noqa: BLE001
        return False                                        # unreadable: not ours


def load_viewer_msr(path, *, prefs: dict | None = None, log=None) -> list:
    """Every MINFLUX dataset in *path*, stamped and with its state restored.

    Returns them in file order, which is the order they were saved in, so the
    overlay indices the state carries line up with the list.
    """
    from ..core.loader import load_from_mfx_array
    from .import_stamp import stamp_msr_dataset
    from .msr_parser import parse_general
    from .viewer_state import apply_viewer_state, read_viewer_state

    source = Path(path)
    parsed = parse_general(str(source), str(source.parent),
                           log=log or (lambda *_a: None))
    datasets = []
    for entry in parsed.get("datasets") or []:
        mfx = entry.get("_mfx")
        if mfx is None or not len(mfx):
            continue
        key = str(entry.get("display_name") or "")
        ds = load_from_mfx_array(
            mfx=mfx,
            name=f"{source.name} | {key}" if key else source.name,
            folder=str(source.parent),
            recent_path=str(source),
            prefs=prefs or {"data": {}},
        )
        stamp_msr_dataset(ds, entry=entry, msr_path=source, key=key,
                          mbm_points=entry.get("_mbm"), log=log)
        # Applied last, so the saved overlay id, LUT and transform win over
        # anything the import would otherwise decide.
        applied = apply_viewer_state(ds, read_viewer_state(entry.get("zroot")))
        if applied and log is not None:
            log(f"[viewer] restored state for '{key}': {', '.join(applied)}")
        datasets.append(ds)
    return datasets
