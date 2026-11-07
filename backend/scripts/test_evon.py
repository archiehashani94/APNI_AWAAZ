"""Call the LLM interface. Without EVON_CHAT_URL this should print canned fallback."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.provider import suggest_with_fallback  # noqa: E402


def main() -> int:
    turns = [
        {"role": "partner", "text": "Aap theek ho? Water chahiye kya?"},
        {"role": "user", "text": "Haan, thoda paani do."},
    ]
    replies, fallback = suggest_with_fallback(turns)
    print("fallback", fallback)
    for item in replies:
        print(item.emotion, item.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
