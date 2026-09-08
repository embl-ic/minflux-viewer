"""Write a ``.msr`` Imspector still opens, by reusing the source file's container.

**Why a transplant and not a synthesised container.** An Imspector ``.msr`` is
not just OBF stacks. Between the 34-byte file header and ``first_stack_position``
sits a ~270 kB UTF-16LE ``<root … oprop_version="1">`` block -- Imspector's
*measurement object*, naming its ExpControl channels. ``msr_reader`` seeks
straight to the first stack and never reads it, so our own writer never knew it
existed; a file without it is refused with *"Failed to read object from file"*
and then *"Unexpected file format"*. It is an Imspector-specific document we
cannot fabricate.

So when a dataset came from a ``.msr``, this keeps that file's bytes -- the
measurement object, every image stack, the file-level ``ome_xml``, the v7 stack
footers and their ``imspector``/``minflux`` tags -- and swaps in only the MFXDTA
payload of the channels being written. Verified in Imspector 16.3 (m2410):
identity, payload-swapped, and payload-swapped-with-viewer-state all open.

⚠ **Two consequences of reusing the container.**

* A dataset with no ``.msr`` origin has no template, and the caller must fall
  back to the minimal writer (which this viewer reads and Imspector does not).
* The output keeps **every** stack of the source, so channels that were not
  loaded are written back with their original payloads. That preserves the file
  rather than silently truncating it, but it does mean the result is the source
  measurement with some channels updated -- not only what was on screen.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

__all__ = ["MAX_DIMS", "template_for", "transplant_payloads", "square_pad"]

MAX_DIMS = 15

# Offsets inside an OBF stack header, after the '<16sLL' magic/version/rank.
_HDR_SIZES = struct.calcsize("<16sLL")                       # res[15]
_HDR_LENS = _HDR_SIZES + MAX_DIMS * 4                        # len[15]
_HDR_TAIL = _HDR_LENS + 2 * MAX_DIMS * 8                     # '<LLLLLQQQ'
_HDR_DATA_LEN = _HDR_TAIL + 4 * 5 + 8
_HDR_NEXT = _HDR_DATA_LEN + 8

# Offsets inside a v7 fixed footer (msr_reader.obffile._read_stack_footer).
_FOOT_STACK_END = 4 + 2 * MAX_DIMS * 4 + 4 + 80 + MAX_DIMS * 80 + 16 + 8   # 1432
_FOOT_END_USED = _FOOT_STACK_END + 8 + 4                                   # 1444
_FOOT_SAMPLES = _FOOT_END_USED + 8                                         # 1452


class TransplantError(ValueError):
    """The source ``.msr`` cannot serve as a template for these datasets."""


def square_pad(payload: bytes) -> tuple[bytes, int]:
    """Pad to a perfect square, the way Imspector stores an MFXDTA blob.

    The vendor declares its MINFLUX stacks 2-D and near-square (3606x3606 for a
    13,003,236-byte payload -- exactly the square). Trailing zeros are safe
    because the MFXDTA record scanner stops after the last record; that is
    already why our reader handles vendor files.
    """
    side = math.ceil(math.sqrt(len(payload))) if payload else 1
    return payload + b"\x00" * (side * side - len(payload)), side


def template_for(datasets) -> Path | None:
    """The one source ``.msr`` all *datasets* came from, or None.

    None whenever they disagree, the path is gone, or a dataset has no ``.msr``
    origin at all -- each of which means there is no container to reuse.
    """
    paths = set()
    for ds in datasets:
        source = str((getattr(ds, "metadata", {}) or {}).get("msr_source_path") or "")
        if not source:
            return None
        paths.add(source)
    if len(paths) != 1:
        return None
    candidate = Path(paths.pop())
    return candidate if candidate.is_file() else None


def _stacks(path: Path):
    from msr_reader.obffile import (
        _read_file_header, _read_stack_footer, _read_stack_header,
    )

    with open(path, "rb") as handle:
        head = _read_file_header(handle)
        found, pos = [], head.first_stack_position
        while pos:
            header = _read_stack_header(handle, pos)
            footer = _read_stack_footer(handle, header)
            found.append((pos, header, footer))
            pos = header.next_stack_position
    return head, found


def transplant_payloads(template: str | Path, payload_for_did: dict,
                        out_path: str | Path) -> dict:
    """Copy *template*, replacing the MFXDTA payload of each DID in the map.

    ``payload_for_did`` maps an Imspector dataset DID to a freshly packed MFXDTA
    container. A DID the file does not contain is reported rather than ignored,
    because silently writing the source unchanged would look like a successful
    save of work that was not saved.
    """
    template = Path(template)
    raw = template.read_bytes()
    head, stacks = _stacks(template)

    # Everything before the first stack, verbatim: the file header AND the
    # Imspector measurement object that makes this a readable measurement.
    pre = bytearray(raw[:head.first_stack_position])
    tail = raw[head.metadata_position:] if head.metadata_position else b""

    pieces, swapped = [], []
    for index, (start, header, footer) in enumerate(stacks):
        end = (stacks[index + 1][0] if index + 1 < len(stacks)
               else (head.metadata_position or len(raw)))
        head_bytes = bytearray(raw[start:header.data_position])
        data = raw[header.data_position:header.data_position + header.data_length]
        trailer = bytearray(raw[header.data_position + header.data_length:end])

        did = None
        tag = (footer.tag_dictionary or {}).get("minflux")
        if tag:
            try:
                did = json.loads(tag).get("did")
            except Exception:                            # noqa: BLE001 - not ours
                did = None
        if did and did in payload_for_did:
            data, side = square_pad(payload_for_did[did])
            struct.pack_into(f"<{MAX_DIMS}L", head_bytes, _HDR_SIZES,
                             *([side, side] + [1] * (MAX_DIMS - 2)))
            struct.pack_into(f"<{MAX_DIMS}d", head_bytes, _HDR_LENS,
                             *([float(side), float(side)] + [1.0] * (MAX_DIMS - 2)))
            struct.pack_into("<Q", head_bytes, _HDR_DATA_LEN, len(data))
            if len(trailer) > _FOOT_SAMPLES + 8:
                struct.pack_into("<Q", trailer, _FOOT_SAMPLES, len(data))
            swapped.append(did)
        pieces.append((head_bytes, data, trailer, start, header.data_length))

    missing = sorted(set(payload_for_did) - set(swapped))
    if missing:
        raise TransplantError(
            f"{template.name} has no MINFLUX dataset for "
            f"{', '.join(missing)} — it is not the file these datasets came from.")

    starts, cursor = [], len(pre)
    for head_bytes, data, trailer, _s, _n in pieces:
        starts.append(cursor)
        cursor += len(head_bytes) + len(data) + len(trailer)
    metadata_position = cursor

    body = bytearray()
    for index, (head_bytes, data, trailer, old_start, old_len) in enumerate(pieces):
        struct.pack_into("<Q", head_bytes, _HDR_NEXT,
                         starts[index + 1] if index + 1 < len(pieces) else 0)
        # ⚠ stack_end_disk is NOT "where the next stack begins" -- Imspector
        # leaves slack after the footer. Recomputing it corrupted 142 bytes
        # across 36 stacks; shift each stack's own value by how far it moved
        # instead, so an untouched stack keeps its bytes exactly.
        delta = (starts[index] - old_start) + (len(data) - old_len)
        if delta and len(trailer) > _FOOT_SAMPLES + 8:
            for offset in (_FOOT_STACK_END, _FOOT_END_USED):
                (old,) = struct.unpack_from("<Q", trailer, offset)
                struct.pack_into("<Q", trailer, offset, old + delta)
        body += bytes(head_bytes) + data + bytes(trailer)

    if head.metadata_position:
        struct.pack_into(
            "<Q", pre,
            struct.calcsize("<10sLQL") + len(head.description.encode("utf-8")),
            metadata_position)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(pre) + bytes(body) + tail)
    return {"path": out_path, "template": template, "swapped": swapped,
            "stacks": len(pieces), "bytes": out_path.stat().st_size}
