"""Local signer/receiver contract, using ephemeral encrypted test keys."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from minflux_viewer.core import signed_updates as signed


@pytest.fixture
def signer():
    path = Path(__file__).resolve().parents[1] / "tools" / "sign_release.py"
    spec = importlib.util.spec_from_file_location("sign_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_encrypted_key_signs_verifiable_metadata(signer, tmp_path, monkeypatch, capsys):
    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "secret.pem"
    key_path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(b"test-password")))
    archive = tmp_path / "app-win.zip"
    archive.write_bytes(b"test archive")
    expiry = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    output = tmp_path / "release"
    monkeypatch.setattr(signer.getpass, "getpass", lambda prompt: "test-password")
    assert signer.main(["--key", str(key_path), "--tag", "v9.9.9", "--expires", expiry,
                        "--asset", "windows", "x86_64", str(archive), "--output-dir", str(output)]) == 0
    pin = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    monkeypatch.setattr(signed, "TRUSTED_RELEASE_KEYS", (pin,))
    manifest = signed.verify_manifest((output / signed.MANIFEST_NAME).read_bytes(),
                                      (output / signed.SIGNATURE_NAME).read_bytes(),
                                      tag="v9.9.9", current_version="0.4.4")
    assert manifest.assets[0].name == archive.name
    assert "test-password" not in capsys.readouterr().out
    assert sorted(p.name for p in output.iterdir()) == [signed.MANIFEST_NAME, signed.SIGNATURE_NAME]


def test_key_inside_checkout_is_refused_before_password_prompt(signer, tmp_path, monkeypatch):
    monkeypatch.setattr(signer.getpass, "getpass", lambda prompt: pytest.fail("password requested"))
    with pytest.raises(SystemExit):
        signer.main(["--key", str(Path(__file__).parent / "never-created.pem"), "--tag", "v9.9.9",
                     "--expires", "2099-01-01T00:00:00Z", "--asset", "windows", "x86_64", "app.zip",
                     "--output-dir", str(tmp_path)])


@pytest.mark.parametrize("system,arch", [("linux", "universal2"), ("other", "x86_64"), ("windows", "other")])
def test_signer_rejects_unsupported_targets(signer, tmp_path, system, arch):
    archive = tmp_path / "app.zip"
    archive.write_bytes(b"a")
    with pytest.raises(ValueError):
        signer.build_manifest("v9.9.9", "2099-01-01T00:00:00Z", [[system, arch, str(archive)]])
