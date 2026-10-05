from __future__ import annotations

from defectlens.data.download import compute_sha256
from defectlens.data.manifest import (
    DataManifest,
    build_manifest,
    read_manifest,
    write_manifest,
)


def test_manifest_roundtrip(tmp_path) -> None:
    payload = b"fake-tar-content" * 100
    archive = tmp_path / "VisA_20220922.tar"
    archive.write_bytes(payload)
    manifest = build_manifest(dataset="visa", archive=archive, expected_size=len(payload))
    assert manifest.sha256 == compute_sha256(archive)
    out = write_manifest(manifest, tmp_path / "visa.manifest.yaml")
    loaded = read_manifest(out)
    assert loaded.dataset == "visa"
    assert loaded.sha256 == manifest.sha256
    assert loaded.size_bytes == manifest.size_bytes


def test_verify_detects_tampered_archive(tmp_path) -> None:
    archive = tmp_path / "data.tar"
    archive.write_bytes(b"original")
    manifest = build_manifest(dataset="x", archive=archive)
    archive.write_bytes(b"tampered-and-longer")
    issues = manifest.verify(tmp_path)
    assert any("size" in i or "sha256" in i for i in issues)


def test_verify_missing_archive(tmp_path) -> None:
    manifest = DataManifest(
        dataset="x", archive="ghost.tar", size_bytes=99, downloaded_at="2026-10-04T00:00:00+00:00"
    )
    issues = manifest.verify(tmp_path)
    assert issues and "missing" in issues[0]
