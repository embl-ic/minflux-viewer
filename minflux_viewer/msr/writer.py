"""
minflux_viewer.msr.writer
=========================
Write MINFLUX localizations (and optional MBM/bead reference data) into an
Imspector-style ``.msr`` file that **our own reader round-trips** — the inverse
of :mod:`minflux_viewer.msr.mfxdta` + ``msr_reader.obffile``.

This is deliberately **"Target A"**: the goal is a file that reopens in
minflux-viewer (``GeneralMSRParser`` → ``load_from_mfx_array``), recovering the
``mfx`` array and any ``grd/mbm/points`` beads. It is **not** trying to be a
byte-perfect Imspector/Abberior file (that would need the closed OmasIo writer
and an Imspector license to validate — see the feasibility notes in
``BACKLOG.md``). We therefore write the **simplest legal** OBF: one 1-D ``uint8``
stack per channel, ``format_version = 1`` (no file-level metadata block), and a
minimal v1 stack footer.

Four nested layers, each the exact inverse of the decoder:

    structured mfx array  (+ optional grd/mbm/points beads)
      └─ zarr v2 store      {key: bytes}          build_channel_store()
           └─ MFXDTA archive  magic + entry table  pack_mfxdta()
                └─ 1-D uint8 OBF stack             \
                     └─ OBF container (.msr/.obf)   } write_obf_file()/obf_bytes()

The two inner layers reuse ``zarr`` (already a dependency); the outer two are a
few ``struct.pack`` calls that mirror ``msr_reader/obffile.py`` byte-for-byte.
"""

from __future__ import annotations

import json
import struct
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mfxdta import MFXDTA_MAGIC

# --- OBF container constants (mirror msr_reader/obffile.py) --------------------
_MAGIC_FILE = b"OMAS_BF\n\xff\xff"            # 10 bytes
_MAGIC_STACK = b"OMAS_BF_STACK\n\xff\xff"     # 16 bytes
_MAX_DIMS = 15                                # MAXIMAL_NUMBER_OF_DIMENSIONS
_DT_UINT8 = 0x00000001                        # OMAS_DT_UINT8

#: Default MFXDTA container version to emit — v3 uses ``/`` key separators
#: (the modern multi-channel layout), which ``extract_zarr_store`` normalises.
_MFXDTA_VERSION = 3
#: zarr subtree prefix inside the MFXDTA archive (``extract_zarr_store`` strips it).
_ZARR_PREFIX = "zarr"

#: Default structured dtype for a ``grd/mbm/points`` bead array.
BEAD_DTYPE = np.dtype([
    ("gri", "<i4"),                # bead group id
    ("xyz", "<f8", (3,)),          # position, metres
    ("tim", "<f8"),                # time, seconds
    ("str", "<f8"),                # signal strength (arbitrary)
])


@dataclass
class MsrChannel:
    """One MINFLUX channel → one OBF stack / MFXDTA store in the ``.msr``."""

    mfx: np.ndarray                                  # 1-D structured m2410 mfx
    name: str = "minflux"
    did: str | None = None                           # UUID; generated if None
    beads: np.ndarray | None = None                  # grd/mbm/points structured array
    points_by_gri: dict | None = None                # {gri: {"name": R-id, "gri": n}}
    used: list | None = None                         # used bead R-ids (e.g. ["R1"])
    acquisition_date: str | None = None              # ISO 8601 instrument timestamp

    def resolved_did(self) -> str:
        return self.did or str(uuid.uuid4())


# =============================================================================
# Layer 1 — structured arrays → in-memory zarr v2 store
# =============================================================================
def build_channel_store(channel: MsrChannel, did: str) -> dict[str, bytes]:
    """Return an in-memory zarr store ``{key: bytes}`` for one channel.

    Contains ``mfx`` (with ``did`` and, when known, ``acquisition_date`` attrs)
    and, when beads are supplied, a
    ``grd/mbm/points`` array (+ ``points_by_gri`` attr) and a root ``mbm`` group
    (+ ``used`` attr) — exactly the three nodes the reader + ``beads_drift``
    consume.
    """
    from . import zarr2

    mfx = np.asarray(channel.mfx)
    if mfx.dtype.names is None:
        raise ValueError("channel.mfx must be a structured array")
    store: dict[str, bytes] = {}
    root = zarr2.open(store, mode="w")
    root["mfx"] = mfx.ravel()
    root["mfx"].attrs["did"] = str(did)
    # The instrument acquisition timestamp, in the same place and format
    # Imspector uses (see core.acquisition_time). Without this a round trip
    # through our writer would silently lose when the data was recorded — the
    # container timestamp below is only the save time.
    if channel.acquisition_date:
        root["mfx"].attrs["acquisition_date"] = str(channel.acquisition_date)

    beads = channel.beads
    if beads is not None and getattr(beads, "size", 0):
        beads = np.asarray(beads)
        if beads.dtype.names is None:
            raise ValueError("channel.beads must be a structured array")
        root["grd/mbm/points"] = beads.ravel()
        if channel.points_by_gri:
            root["grd/mbm/points"].attrs["points_by_gri"] = _jsonable(channel.points_by_gri)
        mbm_grp = root.require_group("mbm")
        if channel.used:
            mbm_grp.attrs["used"] = list(channel.used)
    return store


def _jsonable(obj):
    """Coerce numpy scalars in a nested dict to plain Python for zarr .zattrs."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# =============================================================================
# Layer 2 — zarr store → MFXDTA container bytes
# =============================================================================
def pack_mfxdta(store: dict[str, bytes], *, version: int = _MFXDTA_VERSION,
                timestamp: int | None = None, prefix: str = _ZARR_PREFIX) -> bytes:
    """Pack an in-memory zarr store into an ``MFXDTA`` container.

    Exact inverse of :func:`minflux_viewer.msr.mfxdta.parse_mfxdta_entries` /
    :func:`extract_zarr_store`: header ``MFXDTA\\0`` + ``uint32`` version +
    ``uint32`` timestamp, then one leaf record ``(u64 name_len, name, u8 flag=0,
    u64 data_len, data)`` per store key. Keys are emitted under ``<prefix>/…`` so
    the decoder's ``zarr/`` filter keeps them; the key separator is version-
    dependent (``\\`` for v2, ``/`` for v3+).
    """
    sep = "/" if version >= 3 else "\\"
    ts = int(time.time()) if timestamp is None else int(timestamp)

    out = bytearray()
    out += MFXDTA_MAGIC + b"\x00"                 # bytes 0..6  ("MFXDTA\0")
    out += struct.pack("<I", int(version))        # bytes 7..10  version
    out += struct.pack("<I", ts & 0xFFFFFFFF)     # bytes 11..14 timestamp
    # No padding: the first record begins at offset 15 (the decoder scans from 15).
    for key, data in store.items():
        name = f"{prefix}/{key}"
        if sep != "/":
            name = name.replace("/", sep)
        nb = name.encode("latin1")
        out += struct.pack("<Q", len(nb))
        out += nb
        out += b"\x00"                            # flag 0 = leaf (1 would be a group)
        out += struct.pack("<Q", len(data))
        out += bytes(data)
    return bytes(out)


# =============================================================================
# Layer 3 — MFXDTA payload(s) → OBF container (.msr / .obf)
# =============================================================================
@dataclass
class ObfStack:
    """A single OBF stack to serialise (a 1-D ``uint8`` payload for MINFLUX)."""

    payload: bytes
    name: str = ""
    description: str = ""


def obf_bytes(stacks: list[ObfStack], *, file_description: str = "") -> bytes:
    """Serialise ``stacks`` into a complete OBF (``.msr``/``.obf``) byte string.

    Writes ``format_version = 1`` (so no file-level metadata block is needed) and
    one minimal v1-footer 1-D ``uint8`` stack per entry, linked via
    ``next_stack_pos``. Round-trips through ``msr_reader.obffile.OBFFile``.
    """
    descr = file_description.encode("utf-8")
    # File header: '<10sLQL' magic, format_version, first_stack_pos, descr_len.
    header_fixed = struct.calcsize("<10sLQL")     # 26
    first_stack_pos = header_fixed + len(descr)

    # Pre-serialise each stack body so we know its length → correct next_stack_pos.
    blocks: list[bytes] = []
    positions: list[int] = []
    pos = first_stack_pos
    for st in stacks:
        positions.append(pos)
        pos += len(_stack_block(st, next_stack_pos=0))   # length only (nextpos size-invariant)
    # Second pass with real next_stack_pos now that all starts are known.
    for i, st in enumerate(stacks):
        nxt = positions[i + 1] if i + 1 < len(stacks) else 0
        blocks.append(_stack_block(st, next_stack_pos=nxt))

    out = bytearray()
    out += struct.pack("<10sLQL", _MAGIC_FILE, 1, first_stack_pos if stacks else 0, len(descr))
    out += descr
    for b in blocks:
        out += b
    return bytes(out)


def _stack_block(stack: ObfStack, *, next_stack_pos: int) -> bytes:
    """Serialise one OBF stack: header + name + descr + uint8 data + v1 footer."""
    payload = bytes(stack.payload)
    name = stack.name.encode("utf-8")
    descr = stack.description.encode("utf-8")
    n = len(payload)

    out = bytearray()
    # '<16sLL' magic, version, rank   (version 1 → a v1 footer is present)
    out += struct.pack("<16sLL", _MAGIC_STACK, 1, 1)
    # res[15] uint32 — size along each dim; dim 0 = n, rest 0.
    res = [n] + [0] * (_MAX_DIMS - 1)
    out += struct.pack(f"<{_MAX_DIMS}L", *res)
    # len[15], off[15] doubles — physical length / offset per dim (unused → 0).
    out += struct.pack(f"<{_MAX_DIMS}d", *([0.0] * _MAX_DIMS))
    out += struct.pack(f"<{_MAX_DIMS}d", *([0.0] * _MAX_DIMS))
    # '<LLLLLQQQ' dt, compression_type, compression_level, name_len, descr_len,
    #             reserved, data_len, next_stack_pos.
    out += struct.pack("<LLLLLQQQ", _DT_UINT8, 0, 0, len(name), len(descr),
                       0, n, next_stack_pos)
    out += name
    out += descr
    out += payload
    out += _v1_footer(rank=1)
    return bytes(out)


def _v1_footer(*, rank: int) -> bytes:
    """A minimal OBF v1 stack footer (no SI units / tag dict / flush points).

    Fixed part (``size`` bytes): ``size`` + ``has_col_positions[15]`` +
    ``has_col_labels[15]`` + ``metadata_length`` = 128 bytes; the reader seeks
    ``data_end + size`` to reach the per-dimension label strings, which follow.
    """
    fixed_size = struct.calcsize("<L") + 2 * struct.calcsize(f"<{_MAX_DIMS}L") + struct.calcsize("<L")
    out = bytearray()
    out += struct.pack("<L", fixed_size)                          # footer.size
    out += struct.pack(f"<{_MAX_DIMS}L", *([0] * _MAX_DIMS))      # has_col_positions
    out += struct.pack(f"<{_MAX_DIMS}L", *([0] * _MAX_DIMS))      # has_col_labels
    out += struct.pack("<L", 0)                                   # metadata_length = 0
    # After the fixed footer: one label string per dimension (empty → len 0),
    # then metadata_length(0) bytes of metadata.
    for _ in range(rank):
        out += struct.pack("<L", 0)
    return bytes(out)


# =============================================================================
# High level — channels / datasets → a written .msr file
# =============================================================================
def write_minflux_msr(path, channels: list[MsrChannel], *,
                      timestamp: int | None = None) -> Path:
    """Write ``channels`` to ``path`` as an OBF ``.msr`` our reader round-trips.

    Each channel becomes one MFXDTA-carrying OBF stack. Channel names are
    recorded both as the stack description and (with the DID) in a file-level
    ``[{"did","label"}]`` JSON blob, so ``extract_did_label_map`` recovers them.
    """
    if not channels:
        raise ValueError("write_minflux_msr: no channels to write")

    did_label: list[dict] = []
    stacks: list[ObfStack] = []
    for ch in channels:
        did = ch.resolved_did()
        did_label.append({"did": did, "label": ch.name})
        store = build_channel_store(ch, did)
        blob = pack_mfxdta(store, timestamp=timestamp)
        stacks.append(ObfStack(payload=blob, name=ch.name, description=ch.name))

    file_descr = json.dumps(did_label)
    data = obf_bytes(stacks, file_description=file_descr)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def channel_from_dataset(ds) -> MsrChannel:
    """Build an :class:`MsrChannel` from a viewer dataset.

    The ``mfx`` is the canonical raw array (``dataset_to_mfx_array``); beads are
    read from the ``mbm`` convention used by the MSR loader
    (``metadata["mbm_points"]`` + our ``mbm_points_by_gri`` / ``mbm_used``, or a
    ``dataset.mbm`` ``points`` attr).
    """
    from ..core.save import dataset_to_mfx_array

    mfx = dataset_to_mfx_array(ds)
    meta = getattr(ds, "metadata", {}) or {}
    beads = meta.get("mbm_points")
    if beads is None and getattr(ds, "mbm", None) is not None:
        beads = ds.mbm.get_attr("points")
    return MsrChannel(
        mfx=mfx,
        name=str(getattr(ds, "name", "") or "")
             or getattr(getattr(ds, "file", None), "name", None) or "minflux",
        # ⚠ The MSR reader stamps ``msr_dataset_did``; reading only ``did`` meant
        # ``resolved_did`` minted a fresh UUID on every save, so a re-saved .msr
        # lost the acquisition identity its sidecar matches on -- and the
        # container transplant could never find the stack to replace.
        did=str(meta.get("msr_dataset_did") or meta.get("did") or "") or None,
        beads=beads,
        points_by_gri=meta.get("mbm_points_by_gri"),
        used=meta.get("mbm_used"),
        acquisition_date=str(meta.get("acquisition_date") or "") or None,
    )


def write_datasets_msr(path, datasets, *, timestamp: int | None = None,
                       viewer_state: bool = True, roi_records=None,
                       app_version: str = "", log=None) -> Path:
    """Write one or more viewer datasets (e.g. an overlay group) to a ``.msr``.

    Two containers, chosen by whether a template is available:

    * the datasets all came from one still-present ``.msr`` -> that file's
      container is reused and only the MINFLUX payloads are swapped, so
      **Imspector opens the result** (see :mod:`minflux_viewer.msr.transplant`);
    * otherwise -> the minimal OBF this module writes, which this viewer reads
      and Imspector refuses, because the measurement object it wants cannot be
      synthesised.

    Either way the processing state travels inside each channel's zarr store
    (``viewer/``), so the file is a complete MINFLUX Viewer document.
    """
    from .transplant import TransplantError, template_for, transplant_payloads
    from .viewer_state import build_project_manifest, dataset_viewer_state, write_viewer_state

    datasets = list(datasets)
    manifest = build_project_manifest(datasets) if viewer_state else {}
    by_dataset = _roi_records_by_dataset(roi_records)

    def _report(message: str) -> None:
        if log is not None:
            log(message)

    channels, stores = [], {}
    for index, ds in enumerate(datasets):
        channel = channel_from_dataset(ds)
        did = channel.resolved_did()
        store = build_channel_store(channel, did)
        if viewer_state:
            write_viewer_state(store, dataset_viewer_state(
                ds, manifest=manifest, app_version=app_version,
                roi_records=by_dataset.get(index, by_dataset.get(id(ds), []))))
        channels.append((channel, did))
        stores[did] = store

    template = template_for(datasets)
    if template is not None:
        payloads = {did: pack_mfxdta(store, timestamp=timestamp)
                    for did, store in stores.items()}
        try:
            info = transplant_payloads(template, payloads, path)
        except TransplantError as exc:
            _report(f"[msr] {exc} Falling back to the minimal container, which "
                    "this viewer reads but Imspector does not.")
        else:
            _report(f"[msr] wrote {info['path'].name} from the {template.name} "
                    f"container ({info['stacks']} stacks, "
                    f"{len(info['swapped'])} payload(s) replaced) — "
                    "opens in Imspector.")
            return info["path"]
    else:
        _report("[msr] no source .msr to reuse as a container, so the file "
                "carries only the MINFLUX stacks: this viewer reads it, "
                "Imspector does not.")

    stacks = []
    did_label = []
    for channel, did in channels:
        did_label.append({"did": did, "label": channel.name})
        blob = pack_mfxdta(stores[did], timestamp=timestamp)
        stacks.append(ObfStack(payload=blob, name=channel.name,
                               description=channel.name))
    data = obf_bytes(stacks, file_description=json.dumps(did_label))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def _roi_records_by_dataset(roi_records) -> dict:
    """Group flat ROI records by the dataset they belong to.

    ``save_roi_records`` stamps ``dataset_id`` ("d000000") for a multi-member
    save and omits it for a single one, so both shapes have to work.
    """
    grouped: dict = {}
    for record in roi_records or []:
        raw = str(record.get("dataset_id") or "d000000")
        try:
            index = int(raw.lstrip("d") or 0)
        except ValueError:
            index = 0
        grouped.setdefault(index, []).append(record)
    return grouped
