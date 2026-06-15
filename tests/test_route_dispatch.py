"""Format resolution for routing (extension vs. content, mislabelled files)."""

import io

import numpy as np

from minflux_viewer.core.format_sniff import EXT_TO_FMT, resolve_format


def test_msr_resolves_to_msr(tmp_path):
    # Imspector .msr is OBF; sniff returns None, so the extension is trusted.
    p = tmp_path / "a.msr"
    p.write_bytes(b"OMAS_BF\n" + b"\x00" * 64)
    assert resolve_format(p)[0] == "msr"
    assert EXT_TO_FMT[".msr"] == "msr"


def test_known_extensions_trusted(tmp_path):
    csv = tmp_path / "a.csv"
    csv.write_text("x [nm],y [nm]\n1,2\n", encoding="utf-8")
    assert resolve_format(csv)[0] == "spreadsheet"
    mat = tmp_path / "a.mat"
    mat.write_bytes(b"MATLAB 5.0 MAT-file" + b"\x00" * 64)
    assert resolve_format(mat)[0] == "mat"
    npy = tmp_path / "a.npy"
    np.save(npy, np.zeros(5))
    assert resolve_format(npy)[0] == "npy"


def test_binary_magic_overrides_wrong_extension(tmp_path):
    # A NumPy array mislabelled with a .json extension → loads as npy.
    buf = io.BytesIO(); np.save(buf, np.zeros(5))
    p = tmp_path / "b.json"
    p.write_bytes(buf.getvalue())
    fmt, note = resolve_format(p)
    assert fmt == "npy" and "content is npy" in note


def test_real_json_stays_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('[{"x": 1}]', encoding="utf-8")
    assert resolve_format(p) == ("json", "")


def test_unknown_extension_uses_content(tmp_path):
    # NumPy bytes written to an extension-less path → identified by content.
    buf = io.BytesIO(); np.save(buf, np.zeros(3))
    p = tmp_path / "noext_data"
    p.write_bytes(buf.getvalue())
    fmt, note = resolve_format(p)
    assert fmt == "npy" and note


def test_unidentifiable(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(bytes(range(256)) * 4)          # binary, no magic
    assert resolve_format(p) == (None, "")
