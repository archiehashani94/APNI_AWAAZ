"""Persist cloned-voice embeddings so a backend restart never loses a voice.

Gnani charges for embedding creation, and the user should not have to record
again just because the server restarted. Embeddings are small JSON blobs, so
they are stored under ``backend/data/voices`` keyed by a hash of the browser's
random client id. The directory is gitignored and never served directly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from app.config import BACKEND_DIR

log = logging.getLogger("apni_awaaz.voices")

VOICES_DIR = BACKEND_DIR / "data" / "voices"
_CLIENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _path(client_id: str | None) -> Path | None:
    if not client_id or not _CLIENT_RE.match(client_id):
        return None
    # Match the recordings module: the first 12 characters identify a browser.
    # This lets a saved MP4 and a saved embedding line up even if the client id
    # is shortened in a filename.
    slug = client_id[:12]
    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:32]
    return VOICES_DIR / f"{digest}.json"


def save_voice(client_id: str | None, embedding: dict[str, Any]) -> bool:
    path = _path(client_id)
    if path is None or not embedding:
        return False
    try:
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding": embedding,
            "saved_at": int(time.time() * 1000),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        log.info("saved clone embedding for client %s", client_id[:8])
        return True
    except OSError as exc:
        log.warning("could not persist clone embedding: %s", exc)
        return False


def load_voice(client_id: str | None) -> dict[str, Any] | None:
    path = _path(client_id)
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    embedding = payload.get("embedding")
    if not isinstance(embedding, dict) or "embedding" not in embedding:
        return None
    return embedding
