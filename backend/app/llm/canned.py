from __future__ import annotations

import re

from app.llm.interface import ReplyOption

DEVANAGARI = re.compile(r"[\u0900-\u097F]")
HINGLISH = re.compile(
    r"\b(aap|theek|kya|hai|ho|hain|chahiye|haan|nahi|nahin|paani|dhanyavaad|namaste|thoda|bilkul)\b",
    re.I,
)


def canned_replies(turns: list[dict[str, str]]) -> list[ReplyOption]:
    last = next((t["text"] for t in reversed(turns) if t.get("role") == "partner"), "")
    if DEVANAGARI.search(last) or HINGLISH.search(last):
        return [
            ReplyOption("Haan, theek hai.", "calm"),
            ReplyOption("Dhanyavaad.", "warm"),
            ReplyOption("Mujhe dard ho raha hai. Help chahiye.", "urgent"),
        ]
    return [
        ReplyOption("Yes, that's fine.", "calm"),
        ReplyOption("Thank you.", "warm"),
        ReplyOption("I'm in pain. I need help.", "urgent"),
    ]
