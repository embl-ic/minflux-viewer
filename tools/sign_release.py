"""Sign release assets locally; never run with a private key in GitHub CI.

Example (the encrypted Ed25519 PEM key must live OUTSIDE this checkout)::

    poetry run python tools/sign_release.py --key /secure/release-key.pem \
        --tag v0.5.0 --expires 2026-12-01T00:00:00Z --output-dir /release \
        --asset windows x86_64 /release/app-win.zip \
        --asset linux x86_64 /release/app-linux.tar.gz

Upload release-manifest.json and its raw 64-byte release-manifest.sig alongside
the exact archives. Distribute the printed PUBLIC pin out-of-band in a reviewed
build's TRUSTED_RELEASE_KEYS. No private key is generated or uploaded here.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from minflux_viewer.core.signed_updates import (
    MANIFEST_NAME,
    MAX_ASSETS,
    MAX_MANIFEST_BYTES,
    SIGNATURE_NAME,
    _tag_version,
    _timestamp,
    validate_asset_name,
    validate_asset_target,
)
from minflux_viewer.core.updater import GITHUB_REPO


def build_manifest(tag: str, expires: str, assets: list[list[str]]) -> bytes:
    """Hash explicitly targeted archives; never infer architecture from a name."""
    _tag_version(tag)
    now = datetime.now(timezone.utc)
    expiry = _timestamp(expires)
    if expiry <= now:
        raise ValueError("Expiry must be in the future.")
    records = []
    names: set[str] = set()
    targets: set[tuple[str, str]] = set()
    if not 0 < len(assets) <= MAX_ASSETS:
        raise ValueError("Provide between 1 and 32 assets.")
    for system, arch, filename in assets:
        path = Path(filename)
        name = validate_asset_name(path.name)
        if name in {MANIFEST_NAME, SIGNATURE_NAME}:
            raise ValueError("Metadata files cannot be release archives.")
        validate_asset_target(system, arch)
        if name.casefold() in names or (system, arch) in targets:
            raise ValueError("Duplicate asset name or platform.")
        names.add(name.casefold())
        targets.add((system, arch))
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            while block := source.read(1 << 20):
                size += len(block)
                digest.update(block)
        if not size:
            raise ValueError("Cannot sign an empty archive.")
        records.append(dict(name=name, system=system, arch=arch, size=size, sha256=digest.hexdigest()))
    payload = json.dumps(dict(schema=1, repository=GITHUB_REPO, tag=tag,
                              issued_at=now.isoformat(), expires_at=expiry.isoformat(),
                              assets=records), sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_MANIFEST_BYTES:
        raise ValueError("Manifest exceeds the verifier's size limit.")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key", required=True, type=Path, help="Encrypted Ed25519 PEM outside the checkout")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expires", required=True, help="Explicit UTC expiry (ISO-8601)")
    parser.add_argument("--asset", nargs=3, action="append", required=True, metavar=("SYSTEM", "ARCH", "PATH"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if args.key.resolve().is_relative_to(project_root):
        parser.error("The private key must be outside the checkout, including ignored directories.")
    # No password argument, environment variable or log output: terminal only.
    password = getpass.getpass("Encrypted release-key password: ").encode("utf-8")
    key = serialization.load_pem_private_key(args.key.read_bytes(), password=password)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("The release key must be an encrypted Ed25519 private key.")
    payload = build_manifest(args.tag, args.expires, args.asset)
    signature = key.sign(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: do not silently replace an already-published manifest.
    manifest_path = args.output_dir / MANIFEST_NAME
    signature_path = args.output_dir / SIGNATURE_NAME
    completed = False
    created_manifest = False
    created_signature = False
    try:
        with manifest_path.open("xb") as output:
            created_manifest = True
            with signature_path.open("xb") as signature_output:
                created_signature = True
                output.write(payload)
                signature_output.write(signature)
        completed = True
    finally:
        if not completed:
            if created_manifest:
                manifest_path.unlink(missing_ok=True)
            if created_signature:
                signature_path.unlink(missing_ok=True)
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    print(f"Public key pin (distribute out-of-band): {public.hex()}")
    print("Created release manifest and detached signature. No upload performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
