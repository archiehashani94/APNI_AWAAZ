"""POST /api/v1/tts/voice-clone/embeddings — needs GNANI_API_KEY and a ~5–30s wav."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.gnani.client import gnani  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="WAV/MP3, 5–30 seconds of clean speech")
    args = parser.parse_args()
    if not settings.gnani_api_key:
        print("SKIP: set GNANI_API_KEY")
        return 0
    path = Path(args.audio)
    data = path.read_bytes()
    result = gnani.create_embeddings(data, path.name, "audio/wav")
    print("shape", result.get("shape"), "dtype", result.get("dtype"))
    print("embedding_chars", len(str(result.get("embedding", ""))))
    print("credits", gnani.ledger.snapshot())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
