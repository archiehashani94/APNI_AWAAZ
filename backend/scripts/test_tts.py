"""POST /api/v1/tts/inference with stock Timbre, then clone if --embedding-json is set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.gnani.client import gnani  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Namaste. This is a test of Apni Awaaz.")
    parser.add_argument("--out", default="tts-test.wav")
    parser.add_argument("--embedding-json", help="JSON file with speaker_embedding fields")
    parser.add_argument("--emotion", default="warm")
    args = parser.parse_args()
    if not settings.gnani_api_key:
        print("SKIP: set GNANI_API_KEY")
        return 0
    embedding = None
    if args.embedding_json:
        embedding = json.loads(Path(args.embedding_json).read_text(encoding="utf-8"))
    audio, voice, stock = gnani.synthesize(
        args.text, speaker_embedding=embedding, emotion=args.emotion
    )
    Path(args.out).write_bytes(audio)
    print("wrote", args.out, "bytes", len(audio), "voice", voice, "stock", stock)
    print("credits", gnani.ledger.snapshot())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
