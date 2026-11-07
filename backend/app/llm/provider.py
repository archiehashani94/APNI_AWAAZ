"""The only file to swap when changing the suggestion LLM.

Evon v3.3 is on a closed beta (gnani.ai/evon-waitlist). docs.gnani.ai does not
publish a chat path. If EVON_CHAT_URL is set, we call OpenAI-compatible
POST {EVON_CHAT_URL}/chat/completions with X-API-Key-ID.
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from app.config import settings
from app.llm.canned import canned_replies
from app.llm.interface import ReplyOption, SuggestLLM

log = logging.getLogger("apni_awaaz.llm")

SYSTEM = """You help someone who cannot speak. Given recent conversation turns,
propose exactly 3 short replies they can tap and have spoken aloud.

Match the partner's language, including Hinglish mix if they used it.
Each reply must be one sentence, under 12 words.
Return JSON only:
{"replies":[{"text":"...","emotion":"warm|urgent|playful|calm"}, ...]}
Use a mix of emotions when it fits.
"""


class EvonProvider:
    def suggest(self, turns: list[dict[str, str]]) -> list[ReplyOption]:
        if not settings.evon_chat_url:
            raise RuntimeError("EVON_CHAT_URL is not configured")
        key = settings.evon_api_key or settings.gnani_api_key
        if not key:
            raise RuntimeError("No API key for Evon")
        transcript = "\n".join(f"{t.get('role', 'unknown')}: {t.get('text', '')}" for t in turns[-6:])
        url = settings.evon_chat_url.rstrip("/") + "/chat/completions"
        body = {
            "model": settings.evon_model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": transcript or "Partner just greeted you."},
            ],
            "temperature": 0.6,
            "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = httpx.post(
            url,
            headers={"X-API-Key-ID": key, "Content-Type": "application/json"},
            json=body,
            timeout=30.0,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_replies(content)


def _parse_replies(content: str) -> list[ReplyOption]:
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise ValueError("LLM did not return JSON")
    data = json.loads(match.group(0))
    allowed = {"warm", "urgent", "playful", "calm"}
    out: list[ReplyOption] = []
    for item in data.get("replies", [])[:3]:
        text = str(item.get("text", "")).strip()
        emotion = item.get("emotion", "warm")
        if emotion not in allowed:
            emotion = "warm"
        if text:
            out.append(ReplyOption(text, emotion))
    if len(out) < 3:
        raise ValueError("Need 3 replies")
    return out


def get_llm() -> SuggestLLM:
    return EvonProvider()


def suggest_with_fallback(turns: list[dict[str, str]]) -> tuple[list[ReplyOption], bool]:
    try:
        replies = get_llm().suggest(turns)
        return replies, False
    except Exception as exc:
        log.warning("LLM failed, using canned replies: %s", exc)
        return canned_replies(turns), True
