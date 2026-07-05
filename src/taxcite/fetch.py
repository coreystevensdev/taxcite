"""Download publications into data/raw with content-hash change detection."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from taxcite.manifest import Publication

DATA_DIR = Path("data/raw")


class FetchError(RuntimeError):
    pass


def fetch_publication(pub: Publication, data_dir: Path = DATA_DIR) -> Path:
    """Download a publication PDF, skipping the write when content is unchanged.

    Writes a sidecar .meta.json with the sha256 and fetch timestamp so
    ingest can tell whether the IRS revised a pub since the last run.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / pub.filename
    meta_path = data_dir / f"{pub.pub_id}.meta.json"

    response = httpx.get(pub.url, follow_redirects=True, timeout=60.0)
    if response.status_code != 200:
        raise FetchError(f"{pub.url} returned {response.status_code}")
    if not response.content.startswith(b"%PDF"):
        raise FetchError(f"{pub.url} did not return a PDF")

    digest = hashlib.sha256(response.content).hexdigest()
    previous = _read_meta(meta_path)
    if previous and previous.get("sha256") == digest and target.exists():
        return target

    target.write_bytes(response.content)
    meta_path.write_text(
        json.dumps(
            {
                "pub_id": pub.pub_id,
                "sha256": digest,
                "bytes": len(response.content),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "changed": previous is not None,
            },
            indent=2,
        )
    )
    return target


def _read_meta(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
