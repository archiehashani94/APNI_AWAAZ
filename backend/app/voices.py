"""Persist cloned-voice embeddings so a backend restart never loses a voice.

Gnani charges for embedding creation, and the user should not have to record
again just because the server restarted — or lose an older take when a newer
one is enrolled. Every embedding is therefore stored as its own small JSON file
under ``backend/data/voices``, named by the browser's random client id plus a
voice id. The directory is gitignored and never served directly.

Earlier builds kept one file per client (``<digest>.json``) and overwrote it on
every enrolment. That file is still read, so an existing clone keeps working.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import BACKEND_DIR

log = logging.getLogger("apni_awaaz.voices")

VOICES_DIR = BACKEND_DIR / "data" / "voices"
_CLIENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_VOICE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
LEGACY_ID = "legacy"


def _digest(client_id: str | None) -> str | None:
    if not client_id or not _CLIENT_RE.match(client_id):
        return None
    # Match the recordings module: the first 12 characters identify a browser.
    # This lets a saved MP4 and a saved embedding line up even if the client id
    # is shortened in a filename.
    slug = client_id[:12]
    return hashlib.sha256(slug.encode("utf-8")).hexdigest()[:32]


def _legacy_path(client_id: str | None) -> Path | None:
    digest = _digest(client_id)
    if digest is None:
        return None
    return VOICES_DIR / f"{digest}.json"


def _voice_path(client_id: str | None, voice_id: str | None) -> Path | None:
    digest = _digest(client_id)
    if digest is None or not voice_id or not _VOICE_RE.match(voice_id):
        return None
    return VOICES_DIR / f"{digest}-{voice_id}.json"


def _read(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    embedding = payload.get("embedding")
    if not isinstance(embedding, dict) or "embedding" not in embedding:
        return None
    return payload


def save_voice(
    client_id: str | None,
    embedding: dict[str, Any],
    *,
    label: str | None = None,
    seconds: float | None = None,
    recording: str | None = None,
) -> dict[str, Any] | None:
    """Store a new voice without touching earlier ones. Returns its metadata."""
    digest = _digest(client_id)
    if digest is None or not embedding:
        return None
    voice_id = uuid4().hex[:16]
    path = VOICES_DIR / f"{digest}-{voice_id}.json"
    saved_at = int(time.time() * 1000)
    payload = {
        "id": voice_id,
        "embedding": embedding,
        "saved_at": saved_at,
        "label": label,
        "seconds": round(seconds, 2) if seconds else None,
        "recording": recording or None,
    }
    try:
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        log.info("saved clone embedding %s for client %s", voice_id, (client_id or "")[:8])
    except OSError as exc:
        log.warning("could not persist clone embedding: %s", exc)
        return None
    return {
        "id": voice_id,
        "saved_at": saved_at,
        "seconds": payload["seconds"],
        "recording": payload["recording"],
        "label": label,
    }


def list_voices(client_id: str | None) -> list[dict[str, Any]]:
    """Return this browser's saved voices, newest first (no embeddings)."""
    digest = _digest(client_id)
    if digest is None or not VOICES_DIR.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in VOICES_DIR.glob(f"{digest}*.json"):
        name = path.name
        if name == f"{digest}.json":
            voice_id = LEGACY_ID
        elif name.startswith(f"{digest}-") and name.endswith(".json"):
            voice_id = name[len(digest) + 1 : -5]
        else:
            continue
        if voice_id != LEGACY_ID and not _VOICE_RE.match(voice_id):
            continue
        payload = _read(path)
        if payload is None:
            continue
        saved_at = payload.get("saved_at")
        if not isinstance(saved_at, (int, float)):
            try:
                saved_at = int(path.stat().st_mtime * 1000)
            except OSError:
                saved_at = 0
        items.append(
            {
                "id": voice_id,
                "saved_at": int(saved_at),
                "seconds": payload.get("seconds"),
                "recording": payload.get("recording"),
                "label": payload.get("label"),
            }
        )
    items.sort(key=lambda item: item["saved_at"], reverse=True)
    return items


def _current_path(client_id: str | None) -> Path | None:
    digest = _digest(client_id)
    if digest is None:
        return None
    # Deliberately not ``.json`` so list_voices() never mistakes it for a voice.
    return VOICES_DIR / f"{digest}.current"


def current_voice_id(client_id: str | None) -> str | None:
    """Return the voice this browser last selected, if it still exists."""
    path = _current_path(client_id)
    if path is None or not path.is_file():
        return None
    try:
        voice_id = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if voice_id == LEGACY_ID or _VOICE_RE.match(voice_id or ""):
        return voice_id
    return None


def set_current_voice(client_id: str | None, voice_id: str | None) -> None:
    """Remember a selection so it survives backend and page restarts."""
    path = _current_path(client_id)
    if path is None or not voice_id:
        return
    try:
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(voice_id, encoding="utf-8")
    except OSError as exc:
        log.warning("could not persist the selected voice: %s", exc)


def clear_current_voice(client_id: str | None) -> None:
    """Forget the remembered selection after its voice is deleted."""
    path = _current_path(client_id)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not clear the selected voice: %s", exc)


def delete_voice(client_id: str | None, voice_id: str | None) -> bool:
    """Delete one saved clone. Returns True when a file was removed."""
    path = (
        _legacy_path(client_id)
        if voice_id == LEGACY_ID
        else _voice_path(client_id, voice_id)
    )
    if path is None or not path.is_file():
        return False
    try:
        path.unlink()
    except OSError as exc:
        log.warning("could not delete clone embedding: %s", exc)
        return False
    if current_voice_id(client_id) == voice_id:
        clear_current_voice(client_id)
    log.info("deleted clone embedding %s for client %s", voice_id, (client_id or "")[:8])
    return True


def load_voice(
    client_id: str | None,
    voice_id: str | None = None,
) -> dict[str, Any] | None:
    """Load one voice by id, or the last selected (else newest) voice."""
    if not voice_id:
        voices = list_voices(client_id)
        if not voices:
            return None
        saved = current_voice_id(client_id)
        known = {item["id"] for item in voices}
        voice_id = saved if saved in known else voices[0]["id"]
    path = (
        _legacy_path(client_id)
        if voice_id == LEGACY_ID
        else _voice_path(client_id, voice_id)
    )
    if path is None or not path.is_file():
        return None
    payload = _read(path)
    return payload.get("embedding") if payload else None
