"""POST /stt/v3 Prisma v3.0 REST (short clip). Live WS is exercised by the app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.gnani.client import gnani  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="WAV/MP3 up to 30 seconds")
    args = parser.parse_args()
    if not settings.gnani_api_key:
        print("SKIP: set GNANI_API_KEY")
        return 0
    path = Path(args.audio)
    with path.open("rb") as handle:
        response = httpx.post(
            f"{settings.gnani_base_url}/stt/v3",
            headers={"X-API-Key-ID": settings.gnani_api_key},
            files={"audio_file": (path.name, handle, "audio/wav")},
            data={"model": "prisma-v3.0", "language_code": "indic-auto"},
            timeout=60.0,
        )
    gnani._record("stt-rest", 20.0, headers=response.headers)  # noqa: SLF001
    print(response.status_code, response.text[:1000])
    print("credits", gnani.ledger.snapshot())
    return 0 if response.status_code < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
