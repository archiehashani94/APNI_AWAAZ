r"""Check whether the configured Gnani key authenticates, without spending credits.

Run from the backend directory:

    .\.venv\Scripts\python.exe scripts\check_gnani_key.py

Why this is free: it sends a deliberately invalid TTS model. Gnani validates
the API key first and the request body second, so a valid key returns 400
"model is not supported" (no synthesis, no charge), while an invalid key
returns 401 INVALID_API_KEY. A 403 SCOPE_NOT_ALLOWED means the key is real
but not allowed to use TTS.

The script never prints the key itself: only its length, last 4 characters,
and a SHA-256 prefix.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402


def fingerprint(key: str) -> str:
    return (
        f"length={len(key)} tail={key[-4:]} "
        f"sha256_10={hashlib.sha256(key.encode()).hexdigest()[:10]}"
    )


def extract_error(payload: object, fallback: str) -> str:
    if not isinstance(payload, dict):
        return fallback
    detail = payload.get("detail")
    if isinstance(detail, dict):
        code = detail.get("error_code") or detail.get("code")
        message = detail.get("message") or detail.get("detail")
        if code and message:
            return f"{code}: {message}"
        if code or message:
            return str(code or message)
    if isinstance(detail, str):
        return detail
    if payload.get("message"):
        return str(payload["message"])
    if payload.get("error"):
        return str(payload["error"])
    return fallback


def main() -> int:
    key = settings.gnani_api_key.strip()
    if not key:
        print("No GNANI_API_KEY is configured in backend/.env")
        return 2

    base = settings.gnani_base_url.rstrip("/")
    print(f"Base URL: {base}")
    print(f"Loaded key: {fingerprint(key)}")

    # An invalid model is rejected after authentication, before synthesis.
    body = {
        "text": "hi",
        "voice": "Nalini",
        "model": "__auth_check__",
        "audio_config": {
            "sample_rate": 8000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "wav",
        },
    }
    url = f"{base}/api/v1/tts/inference"
    try:
        response = httpx.post(
            url,
            headers={"X-API-Key-ID": key, "Content-Type": "application/json"},
            json=body,
            timeout=45.0,
        )
    except httpx.HTTPError as exc:
        print(f"NETWORK ERROR: {type(exc).__name__}: {exc}")
        print("The key was not tested. Check internet access / proxy / firewall.")
        return 3

    try:
        payload: object = response.json()
    except ValueError:
        payload = response.text[:500]
    detail = extract_error(payload, response.text[:500])
    print(f"HTTP {response.status_code}: {detail}")

    if response.status_code == 400 and "model" in detail.lower():
        print("RESULT: KEY IS VALID (request reached body validation).")
        return 0
    if response.status_code == 401:
        print("RESULT: KEY IS INVALID (Gnani does not recognize this value).")
        print("Fix: create/copy the key again, update backend/.env, restart uvicorn.")
        return 1
    if response.status_code == 403:
        print("RESULT: KEY IS VALID BUT LACKS THE TTS SCOPE.")
        print("Fix: enable Text-to-Speech / Voice Cloning for this key in the dashboard.")
        return 1
    if response.status_code == 429:
        print("RESULT: key passed authentication, but the account is rate-limited.")
        return 0
    print("RESULT: could not classify; see the HTTP status above.")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        sys.exit(130)
