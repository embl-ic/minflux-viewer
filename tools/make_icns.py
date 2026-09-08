"""Build ``resources/icons/minflux_viewer_logo.icns`` from the Windows ``.ico``.

A macOS ``.app`` wants an ``.icns``; without one PyInstaller gives the bundle a
generic icon. ``iconutil`` only exists on macOS and Pillow is deliberately not a
dependency of this project, so this writes the container directly: an ICNS is a
short header followed by typed entries, and every modern entry type takes an
ordinary PNG payload.

The source ``.ico`` holds one 256x256 32-bit BGRA image (bottom-up, with a
trailing 1-bit AND mask that is redundant because the alpha channel is present).
The smaller sizes are exact 2:1 box reductions of it -- powers of two all the
way down, so no resampling filter is needed -- averaged with premultiplied
alpha so transparent edge pixels do not drag colour into their neighbours.

Run it only when the logo changes; the result is committed.

    python tools/make_icns.py
"""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ICO = REPO_ROOT / "resources" / "icons" / "minflux_viewer_logo.ico"
DEFAULT_ICNS = REPO_ROOT / "resources" / "icons" / "minflux_viewer_logo.icns"

#: ICNS entry type per square edge length. Every one of these takes a PNG.
ICNS_TYPES = {32: b"ic11", 64: b"ic12", 128: b"ic07", 256: b"ic08"}


class IconError(RuntimeError):
    """The source icon is not in the one layout this converter understands."""


def read_ico_rgba(path: Path) -> tuple[int, bytearray]:
    """The largest 32-bit BGRA frame of *path* as top-down RGBA bytes."""
    raw = path.read_bytes()
    _reserved, image_type, count = struct.unpack("<HHH", raw[:6])
    if image_type != 1 or count < 1:
        raise IconError(f"{path} is not an icon file")

    best: tuple[int, int, int] | None = None          # (edge, offset, size)
    for index in range(count):
        entry = raw[6 + index * 16:22 + index * 16]
        width, height, _colors, _r, _planes, bpp, size, offset = struct.unpack(
            "<BBBBHHII", entry)
        edge = width or 256
        if edge != (height or 256):
            continue                                   # not square: unusable
        if bpp not in (0, 32):
            continue
        if best is None or edge > best[0]:
            best = (edge, offset, size)
    if best is None:
        raise IconError(f"{path} has no square 32-bit frame")

    edge, offset, size = best
    frame = raw[offset:offset + size]
    if frame[:4] == b"\x89PNG":
        raise IconError(
            f"{path} stores its {edge}px frame as PNG; embed it directly "
            "instead of decoding a DIB")

    header_size = struct.unpack("<I", frame[:4])[0]
    dib_width, dib_height, _planes, dib_bpp = struct.unpack(
        "<iiHH", frame[4:16])
    if dib_bpp != 32:
        raise IconError(f"{path}: expected a 32-bit frame, found {dib_bpp}-bit")
    # An icon DIB declares double height: the XOR image plus the AND mask.
    if dib_width != edge or dib_height != edge * 2:
        raise IconError(
            f"{path}: unexpected DIB {dib_width}x{dib_height} for a {edge}px frame")

    pixels = frame[header_size:header_size + edge * edge * 4]
    if len(pixels) != edge * edge * 4:
        raise IconError(f"{path}: truncated pixel data")

    rgba = bytearray(edge * edge * 4)
    for y in range(edge):
        src = (edge - 1 - y) * edge * 4                # DIB rows are bottom-up
        dst = y * edge * 4
        for x in range(0, edge * 4, 4):
            b, g, r, a = pixels[src + x:src + x + 4]
            rgba[dst + x:dst + x + 4] = bytes((r, g, b, a))
    return edge, rgba


def halve(edge: int, rgba: bytes) -> tuple[int, bytearray]:
    """Exact 2:1 box reduction, averaged with premultiplied alpha."""
    half = edge // 2
    out = bytearray(half * half * 4)
    for y in range(half):
        for x in range(half):
            r = g = b = a = 0
            for dy in (0, 1):
                row = ((y * 2 + dy) * edge + x * 2) * 4
                for dx in (0, 4):
                    pr, pg, pb, pa = rgba[row + dx:row + dx + 4]
                    r += pr * pa
                    g += pg * pa
                    b += pb * pa
                    a += pa
            i = (y * half + x) * 4
            if a:
                out[i:i + 4] = bytes((r // a, g // a, b // a, a // 4))
            # else: fully transparent, and the zeros already say so
    return half, out


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def encode_png(edge: int, rgba: bytes) -> bytes:
    """8-bit RGBA PNG, filter 0 on every scanline."""
    lines = bytearray()
    for y in range(edge):
        lines.append(0)
        lines += rgba[y * edge * 4:(y + 1) * edge * 4]
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", edge, edge, 8, 6, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(bytes(lines), 9))
            + _chunk(b"IEND", b""))


def build_icns(ico_path: Path) -> bytes:
    edge, rgba = read_ico_rgba(ico_path)
    images = {edge: bytes(rgba)}
    while edge > min(ICNS_TYPES) and edge % 2 == 0:
        edge, rgba = halve(edge, rgba)
        images[edge] = bytes(rgba)

    entries = b""
    for size in sorted(ICNS_TYPES):
        if size not in images:
            continue
        png = encode_png(size, images[size])
        entries += ICNS_TYPES[size] + struct.pack(">I", len(png) + 8) + png
    if not entries:
        raise IconError(f"{ico_path} produced no usable icon sizes")
    return b"icns" + struct.pack(">I", len(entries) + 8) + entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ico", type=Path, default=DEFAULT_ICO)
    parser.add_argument("--out", type=Path, default=DEFAULT_ICNS)
    args = parser.parse_args()
    try:
        data = build_icns(args.ico)
    except IconError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
