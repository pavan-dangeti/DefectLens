from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

_CHUNK = 1024 * 1024


def compute_sha256(path: Path, chunk_size: int = _CHUNK) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def download(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    timeout: float = 300.0,
    client: httpx.Client | None = None,
) -> Path:
    """Stream ``url`` to ``dest``, verify size and SHA-256, and rename atomically on success.

    On any failure the partial file is removed and the destination is never left half-written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    close_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=timeout)
    try:
        with http.stream("GET", url) as resp:
            resp.raise_for_status()
            real_size = 0
            h = hashlib.sha256()
            with tmp.open("wb") as out:
                for chunk in resp.iter_bytes(_CHUNK):
                    h.update(chunk)
                    out.write(chunk)
                    real_size += len(chunk)
        if expected_size is not None and real_size != expected_size:
            tmp.unlink(missing_ok=True)
            raise IOError(f"size mismatch: got {real_size}, expected {expected_size}")
        digest = h.hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise IOError(f"sha256 mismatch: got {digest}, expected {expected_sha256}")
        tmp.replace(dest)
        return dest
    finally:
        if close_client:
            http.close()


def download_archive(
    url: str, dest_dir: Path, filename: str, expected_size: int | None = None
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    return download(url, dest_dir / filename, expected_size=expected_size)
