from __future__ import annotations

import hashlib

import httpx
import pytest

from defectlens.data.download import download

_PAYLOAD = b"defect-lens-test-payload" * 10_000
_PAYLOAD_SHA = hashlib.sha256(_PAYLOAD).hexdigest()


def _client(payload: bytes = _PAYLOAD) -> httpx.Client:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=payload))
    return httpx.Client(transport=transport, follow_redirects=True)


def test_download_success(tmp_path) -> None:
    dest = download("https://example.test/data.tar", tmp_path / "data.tar", client=_client())
    assert dest.read_bytes() == _PAYLOAD


def test_download_size_mismatch_leaves_no_file(tmp_path) -> None:
    with pytest.raises(IOError, match="size mismatch"):
        download(
            "https://example.test/data.tar",
            tmp_path / "data.tar",
            expected_size=len(_PAYLOAD) + 1,
            client=_client(),
        )
    assert not (tmp_path / "data.tar").exists()
    assert not (tmp_path / "data.tar.part").exists()


def test_download_hash_mismatch_leaves_no_file(tmp_path) -> None:
    with pytest.raises(IOError, match="sha256 mismatch"):
        download(
            "https://example.test/data.tar",
            tmp_path / "data.tar",
            expected_sha256="0" * 64,
            client=_client(),
        )
    assert not (tmp_path / "data.tar").exists()


def test_download_hash_verifies_present(tmp_path) -> None:
    dest = download(
        "https://example.test/data.tar",
        tmp_path / "data.tar",
        expected_sha256=_PAYLOAD_SHA,
        client=_client(),
    )
    assert dest.exists()


def test_http_error_status(tmp_path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(404))
    client = httpx.Client(transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        download("https://example.test/missing.tar", tmp_path / "m.tar", client=client)
    assert not (tmp_path / "m.tar").exists()
    assert not (tmp_path / "m.tar.part").exists()
