"""Persist enrolled voice references as MP4 (AAC) files.

The browser sends the cleaned 16 kHz WAV reference. We keep a playable MP4
copy next to the backend so every recording survives a page reload or a
session loss. FFmpeg comes from ``imageio-ffmpeg``, which ships a static
binary, so no system FFmpeg install is required.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import BACKEND_DIR

log = logging.getLogger("apni_awaaz.recordings")

RECORDINGS_DIR = BACKEND_DIR / "recordings"
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.mp4$")
_CLIENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RecordingError(RuntimeError):
    """Raised when a recording cannot be converted or written."""


def _client_slug(client_id: str | None) -> str:
    if client_id and _CLIENT_RE.match(client_id):
        return client_id[:12]
    return "local"


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def save_recording(
    payload: bytes,
    *,
    client_id: str | None,
    duration: float | None = None,
) -> dict[str, Any]:
    """Convert ``payload`` to MP4/AAC and return its metadata."""
    if not payload:
        raise RecordingError("Empty recording")

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    stem = f"voice-{_client_slug(client_id)}-{stamp}-{uuid4().hex[:6]}"
    target = RECORDINGS_DIR / f"{stem}.mp4"

    with tempfile.TemporaryDirectory(dir=RECORDINGS_DIR) as temp_dir:
        source = Path(temp_dir) / "source.wav"
        source.write_bytes(payload)
        command = [
            _ffmpeg(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vn",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(target),
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120
        )

    if result.returncode != 0 or not target.is_file():
        target.unlink(missing_ok=True)
        detail = (result.stderr or "FFmpeg could not convert the recording").strip()
        raise RecordingError(detail[-400:])

    created = int(time.time() * 1000)
    meta = {
        "name": target.name,
        "url": f"/api/recordings/{target.name}",
        "seconds": round(duration, 2) if duration else None,
        "created_at": created,
        "bytes": target.stat().st_size,
    }
    target.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    log.info("saved recording %s (%.1f kB)", target.name, meta["bytes"] / 1024)
    return meta


def list_recordings(client_id: str | None) -> list[dict[str, Any]]:
    """Return this browser's saved recordings, newest first."""
    if not RECORDINGS_DIR.is_dir():
        return []
    prefix = f"voice-{_client_slug(client_id)}-"
    items: list[dict[str, Any]] = []
    for path in RECORDINGS_DIR.glob("*.mp4"):
        if not path.name.startswith(prefix):
            continue
        meta: dict[str, Any] = {}
        try:
            meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "url": f"/api/recordings/{path.name}",
                "seconds": meta.get("seconds"),
                "created_at": meta.get("created_at") or int(stat.st_mtime * 1000),
                "bytes": stat.st_size,
            }
        )
    items.sort(key=lambda item: item["created_at"], reverse=True)
    return items


def resolve_recording(name: str) -> Path | None:
    """Return a safe path for a recording name, or ``None``."""
    if not _NAME_RE.match(name or ""):
        return None
    directory = RECORDINGS_DIR.resolve()
    path = (directory / name).resolve()
    if path.parent != directory or not path.is_file():
        return None
    return path
