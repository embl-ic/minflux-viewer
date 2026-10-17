"""Release authentication uses a pinned test key, never production credentials."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from minflux_viewer.core import signed_updates as signed

NOW = datetime(2026, 10, 17, 12, tzinfo=timezone.utc)
PAYLOAD = b"an archive"


@pytest.fixture
def signing(monkeypatch):
    key = Ed25519PrivateKey.generate()
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    monkeypatch.setattr(signed, "TRUSTED_RELEASE_KEYS", (public,))
    return key


def manifest(**overrides):
    return {
        "schema": 1,
        "repository": "embl-ic/minflux-viewer",
        "tag": "v9.9.9",
        "issued_at": (NOW - timedelta(hours=1)).isoformat(),
        "expires_at": (NOW + timedelta(days=7)).isoformat(),
        "assets": [{"name": "app-win.zip", "system": "windows", "arch": "x86_64",
                    "size": len(PAYLOAD), "sha256": hashlib.sha256(PAYLOAD).hexdigest()}],
        **overrides,
    }


def encode(value):
    return json.dumps(value, separators=(",", ":")).encode()


def verify(key, data=None, **kwargs):
    raw = encode(manifest() if data is None else data)
    return signed.verify_manifest(raw, key.sign(raw), tag="v9.9.9",
                                  current_version="0.4.4", now=NOW, **kwargs)


def test_valid_manifest_and_exact_platform_selection(signing):
    result = verify(signing)
    assert result.tag == "v9.9.9"
    asset = signed.select_signed_asset(result, system="Windows", machine="AMD64")
    assert asset.name == "app-win.zip"
    with pytest.raises(ValueError, match="platform"):
        signed.select_signed_asset(result, system="Linux", machine="x86_64")


@pytest.mark.parametrize("machine", ["x86_64", "arm64", "aarch64"])
def test_macos_universal_fallback_is_explicit(signing, machine):
    data = manifest()
    data["assets"][0].update(system="darwin", arch="universal2")
    result = verify(signing, data)
    assert signed.select_signed_asset(result, system="Darwin", machine=machine).arch == "universal2"
    with pytest.raises(ValueError, match="platform"):
        signed.select_signed_asset(result, system="Darwin", machine="ppc")


def test_no_trust_is_fail_closed_before_network(monkeypatch, tmp_path):
    monkeypatch.setattr(signed, "TRUSTED_RELEASE_KEYS", ())
    monkeypatch.setattr(signed.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("network without a trust root"))
    assert not signed.trust_configured()
    with pytest.raises(ValueError, match="trusted"):
        signed.download_release("v9.9.9", "0.4.4")


@pytest.mark.parametrize("attack", ["modified", "wrong-key", "short", "long", "no-key"])
def test_signature_must_verify_before_json_parsing(signing, monkeypatch, attack):
    raw = b"not JSON"
    signature = signing.sign(raw)
    if attack == "modified":
        raw += b"!"
    elif attack == "wrong-key":
        signature = Ed25519PrivateKey.generate().sign(raw)
    elif attack == "short":
        signature = signature[:-1]
    elif attack == "long":
        signature += b"!"
    else:
        monkeypatch.setattr(signed, "TRUSTED_RELEASE_KEYS", ())
    monkeypatch.setattr(signed.json, "loads", lambda *a, **k: pytest.fail("parsed untrusted bytes"))
    with pytest.raises(ValueError):
        signed.verify_manifest(raw, signature, tag="v9.9.9", current_version="0.4.4", now=NOW)


@pytest.mark.parametrize("overrides", [
    {"schema": 2}, {"schema": True}, {"repository": "someone/else"},
    {"tag": "v9.9.8"}, {"tag": "v9.9.9/../x"}, {"extra": 1},
    {"expires_at": NOW.isoformat()},
    {"expires_at": (NOW - timedelta(days=1)).isoformat()},
    {"issued_at": (NOW + timedelta(hours=1)).isoformat()},
    {"issued_at": "2026-10-17T00:00:00"},
    {"issued_at": "bad"}, {"assets": []}, {"assets": {}},
])
def test_manifest_rejects_wrong_binding_schema_and_time(signing, overrides):
    with pytest.raises(ValueError):
        verify(signing, manifest(**overrides))


@pytest.mark.parametrize("version", ["9.9.9", "10.0.0", "invalid"])
def test_manifest_cannot_install_same_older_or_uncomparable_version(signing, version):
    raw = encode(manifest())
    with pytest.raises(ValueError):
        signed.verify_manifest(raw, signing.sign(raw), tag="v9.9.9",
                               current_version=version, now=NOW)


@pytest.mark.parametrize("field,value", [
    ("name", "../evil.zip"), ("name", "C:\\evil.zip"), ("name", "/evil.zip"),
    ("name", "evil?x.zip"), ("name", "con.zip"), ("name", ".hidden.zip"),
    ("size", 0), ("size", -1), ("size", True), ("size", "10"),
    ("sha256", "a" * 63), ("sha256", "g" * 64),
    ("system", "other"), ("arch", "other"), ("arch", "universal2"),
])
def test_invalid_asset_records_are_refused(signing, field, value):
    data = manifest()
    data["assets"][0][field] = value
    with pytest.raises(ValueError):
        verify(signing, data)


def test_duplicate_assets_platforms_and_json_keys_are_refused(signing):
    data = manifest()
    data["assets"] *= 2
    with pytest.raises(ValueError):
        verify(signing, data)
    data["assets"][1] = {**data["assets"][0], "name": "second.zip"}
    with pytest.raises(ValueError):
        verify(signing, data)
    raw = encode(manifest()).replace(b'"schema":1', b'"schema":1,"schema":1')
    with pytest.raises(ValueError, match="Duplicate"):
        signed.verify_manifest(raw, signing.sign(raw), tag="v9.9.9",
                               current_version="0.4.4", now=NOW)


def test_manifest_size_is_bounded(signing):
    raw = b" " * (signed.MAX_MANIFEST_BYTES + 1)
    with pytest.raises(ValueError, match="size"):
        signed.verify_manifest(raw, signing.sign(raw), tag="v9.9.9",
                               current_version="0.4.4", now=NOW)


def fake_network(monkeypatch, signing, *, archive=PAYLOAD, raw=None):
    raw = encode(manifest()) if raw is None else raw
    responses = {signed.MANIFEST_NAME: raw, signed.SIGNATURE_NAME: signing.sign(raw),
                 "app-win.zip": archive}
    urls = []

    def open_url(request, **kwargs):
        urls.append(request.full_url)
        name = request.full_url.rsplit("/", 1)[-1]
        response = io.BytesIO(responses[name])
        response.headers = {}  # deliberately omit Content-Length
        return response

    monkeypatch.setattr(signed.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(signed, "_utc_now", lambda: NOW)
    monkeypatch.setattr(signed.platform, "system", lambda: "Windows")
    monkeypatch.setattr(signed.platform, "machine", lambda: "AMD64")
    return urls


def test_download_uses_only_canonical_urls_and_signed_hash(signing, monkeypatch):
    urls = fake_network(monkeypatch, signing)
    download = signed.download_release("v9.9.9", "0.4.4")
    try:
        assert download.archive_path.read_bytes() == PAYLOAD
        assert len(urls) == 3
        assert all(u.startswith("https://github.com/embl-ic/minflux-viewer/releases/download/v9.9.9/")
                   for u in urls)
    finally:
        download.cleanup()
    assert not download.archive_path.exists()


@pytest.mark.parametrize("archive", [b"short", PAYLOAD + b"excess", b"x" * len(PAYLOAD)])
def test_invalid_download_is_removed(signing, monkeypatch, tmp_path, archive):
    fake_network(monkeypatch, signing, archive=archive)
    monkeypatch.setattr(signed.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(ValueError):
        signed.download_release("v9.9.9", "0.4.4")
    assert not list(tmp_path.iterdir())


def test_install_reverifies_archive_and_expiry_before_staging(signing, monkeypatch):
    fake_network(monkeypatch, signing)
    monkeypatch.setattr(signed.self_update, "prepare_update",
                        lambda *a, **k: pytest.fail("staged unverified archive"))
    download = signed.download_release("v9.9.9", "0.4.4")
    try:
        download.archive_path.write_bytes(b"x" * len(PAYLOAD))
        with pytest.raises(ValueError, match="checksum"):
            signed.prepare_download(download)
        download.archive_path.write_bytes(PAYLOAD)
        monkeypatch.setattr(signed, "_utc_now", lambda: NOW + timedelta(days=8))
        with pytest.raises(ValueError, match="expired"):
            signed.prepare_download(download)
    finally:
        download.cleanup()


def test_pin_rotation_accepts_only_explicitly_pinned_keys(signing, monkeypatch):
    other = Ed25519PrivateKey.generate()
    other_public = other.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    monkeypatch.setattr(signed, "TRUSTED_RELEASE_KEYS", (other_public, *signed.TRUSTED_RELEASE_KEYS))
    assert verify(signing).tag == "v9.9.9"
    monkeypatch.setattr(signed, "TRUSTED_RELEASE_KEYS", (other_public,))
    with pytest.raises(ValueError, match="signature"):
        verify(signing)


@pytest.mark.parametrize("pin", ["", "00", "z" * 64, None])
def test_bad_pin_configuration_is_fail_closed(monkeypatch, pin):
    monkeypatch.setattr(signed, "TRUSTED_RELEASE_KEYS", (pin,))
    assert not signed.trust_configured()


def test_oversized_metadata_stops_before_archive_network(signing, monkeypatch):
    raw = b" " * (signed.MAX_MANIFEST_BYTES + 1)
    urls = fake_network(monkeypatch, signing, raw=raw)
    with pytest.raises(ValueError, match="size"):
        signed.download_release("v9.9.9", "0.4.4")
    assert len(urls) == 1


def test_download_cancellation_removes_partial_data(signing, monkeypatch, tmp_path):
    fake_network(monkeypatch, signing)
    monkeypatch.setattr(signed.tempfile, "tempdir", str(tmp_path))

    class Cancelled(Exception):
        pass

    def cancel(done, total):
        if done:
            raise Cancelled

    with pytest.raises(Cancelled):
        signed.download_release("v9.9.9", "0.4.4", progress=cancel)
    assert not list(tmp_path.iterdir())


def test_reverification_cancellation_never_stages(signing, monkeypatch):
    fake_network(monkeypatch, signing)
    download = signed.download_release("v9.9.9", "0.4.4")

    class Cancelled(Exception):
        pass

    def checkpoint():
        raise Cancelled

    monkeypatch.setattr(signed.self_update, "prepare_update", lambda *a: pytest.fail("staged"))
    try:
        with pytest.raises(Cancelled):
            signed.prepare_download(download, checkpoint=checkpoint)
    finally:
        download.cleanup()


def test_signed_download_is_reverified_then_prepared_without_launch(signing, monkeypatch, tmp_path):
    import zipfile

    path = tmp_path / "build.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("minflux_viewer.exe", b"test executable")
    data = manifest()
    payload = path.read_bytes()
    data["assets"][0].update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    fake_network(monkeypatch, signing, archive=payload, raw=encode(data))
    monkeypatch.setattr(signed.self_update, "current_exe", lambda: tmp_path / "install" / "minflux_viewer.exe")
    monkeypatch.setattr(signed.self_update, "is_windows", lambda: True)
    monkeypatch.setattr(signed.self_update, "launch_helper", lambda *a: pytest.fail("launched in worker"))
    download = signed.download_release("v9.9.9", "0.4.4")
    prepared = None
    try:
        prepared = signed.prepare_download(download)
        assert prepared.script_path.is_file()
        assert (prepared.extract_dir / "payload" / "minflux_viewer.exe").read_bytes() == b"test executable"
        assert prepared.script_path.parent == prepared.extract_dir
        assert not (prepared.extract_dir / "payload" / prepared.script_path.name).exists()
    finally:
        download.cleanup()
        if prepared is not None:
            signed.self_update.discard_prepared_update(prepared)
    assert not prepared.extract_dir.exists()
