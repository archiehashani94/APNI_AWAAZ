"""The only file to swap when changing the suggestion LLM.

Reads LLM_PROVIDER and LLM_API_KEY. Empty / 'canned' / missing config uses
the canned reply list — this path must never produce HTTP 503.
"""

from __future__ import annotations

import json
import logging
import re
import time
from threading import Lock
from typing import Any

import httpx

from app.config import settings
from app.llm.canned import canned_replies
from app.llm.interface import ReplyOption, SuggestLLM

log = logging.getLogger("apni_awaaz.llm")

SYSTEM = """You are the voice of a person who cannot speak. They use this app to answer
people around them, and your reply will be spoken aloud in their cloned voice.

You receive the recent conversation. "Partner" is the other person. "You" is
something the user already said.

Write exactly 3 DIFFERENT replies the user could give to the Partner's LAST line.

How to write them:
- Answer what was actually asked. For a yes/no question, at least one reply must
  start with a clear yes or no.
- Be specific to the Partner's words. Never use generic filler that could fit any
  conversation.
- Spread the intents: one direct answer, one clarifying question or request to
  repeat when the partner was unclear, and one need, feeling, or preference.
- Speak in first person as the user. Do not describe actions or add notes.
- Match the partner's language and script exactly: Devanagari Hindi -> Devanagari,
  Hinglish -> Latin Hinglish, English -> English, and so on.
- One short sentence, at most 12 words. Natural spoken language, no lists.
- Never invent facts such as names, times, medicines, or promises.
- Choose an emotion from: warm, urgent, playful, calm.

Return JSON only, with no markdown:
{"replies":[{"text":"...","emotion":"warm"},{"text":"...","emotion":"calm"},{"text":"...","emotion":"warm"}]}
"""

CANNED_PROVIDERS = {"", "canned", "none", "off", "fallback"}
ALLOWED_EMOTIONS = {"warm", "urgent", "playful", "calm"}

# Identical conversation context should not spend another LLM request. This
# also protects the free-tier rate limit when transcripts arrive twice.
_CACHE: dict[str, tuple[float, list[ReplyOption]]] = {}
_CACHE_TTL_SECONDS = 90.0
_CACHE_LOCK = Lock()


def _format_transcript(
    turns: list[dict[str, str]],
    user_gender: str | None = None,
) -> str:
    lines: list[str] = []
    for turn in turns[-8:]:
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        role = "Partner" if turn.get("role") == "partner" else "You"
        lines.append(f"{role}: {text}")
    if not lines:
        body = "Partner just greeted you. Write 3 natural greeting replies."
    else:
        body = (
            "Conversation so far:\n"
            + "\n".join(lines)
            + "\n\nWrite the user's 3 replies to the Partner's last line."
        )
    if user_gender == "female":
        body += (
            "\nThe user is female. Use feminine first-person Hindi/Hinglish verb forms"
            " (for example \"main kar sakti hoon\")."
        )
    elif user_gender == "male":
        body += (
            "\nThe user is male. Use masculine first-person Hindi/Hinglish verb forms"
            " (for example \"main kar sakta hoon\")."
        )
    else:
        body += (
            "\nThe user's gender is unknown. Do not guess it; prefer wording that"
            " avoids gendered first-person verbs when it sounds natural."
        )
    return body


class ChatCompletionsProvider:
    def suggest(
        self,
        turns: list[dict[str, str]],
        user_gender: str | None = None,
    ) -> list[ReplyOption]:
        url = (settings.llm_chat_url or "").rstrip("/")
        if not url:
            raise RuntimeError("LLM_CHAT_URL is not configured")
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        key = settings.llm_api_key
        if not key:
            raise RuntimeError("LLM_API_KEY is not configured")

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _format_transcript(turns, user_gender)},
        ]
        base: dict[str, Any] = {
            "model": settings.llm_model or "gpt-4o-mini",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1000,
        }
        # Gemini 2.5 Flash spends most of max_tokens on hidden "thinking" and
        # truncates the JSON (finish_reason=length). Providers that do not know
        # these fields reject the body, so retry with fewer options.
        variants: list[dict[str, Any]] = [
            {
                "reasoning_effort": "none",
                "response_format": {"type": "json_object"},
            },
            {"response_format": {"type": "json_object"}},
            {},
        ]
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        response: httpx.Response | None = None
        for index, extra in enumerate(variants):
            body = {**base, **extra}
            log.info("LLM POST %s provider=%s attempt=%s", url, settings.llm_provider, index + 1)
            response = httpx.post(url, headers=headers, json=body, timeout=60.0)
            if response.status_code < 400:
                break
            if response.status_code != 400 or index == len(variants) - 1:
                break
            log.info(
                "LLM rejected optional params (status=%s); retrying without them",
                response.status_code,
            )

        assert response is not None
        if response.status_code >= 400:
            log.error(
                "LLM upstream failed status=%s body_chars=%s",
                response.status_code,
                len(response.text),
            )
            response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not content:
            raise ValueError("LLM returned empty content")
        return _parse_replies(content)


def _extract_json(content: str) -> Any:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM did not return JSON")


def _reply_items(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        for key in ("replies", "options", "suggestions", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []
    if isinstance(payload, list):
        return payload
    return []


def _parse_replies(content: str) -> list[ReplyOption]:
    payload = _extract_json(content)
    out: list[ReplyOption] = []
    for item in _reply_items(payload)[:3]:
        if isinstance(item, str):
            text = item.strip()
            emotion = "warm"
        elif isinstance(item, dict):
            text = str(
                item.get("text")
                or item.get("reply")
                or item.get("sentence")
                or ""
            ).strip()
            emotion = item.get("emotion", "warm")
        else:
            continue
        if emotion not in ALLOWED_EMOTIONS:
            emotion = "warm"
        if text:
            out.append(ReplyOption(text, emotion))
    if not out:
        raise ValueError("LLM returned no usable replies")
    return out


def get_llm() -> SuggestLLM | None:
    provider = (settings.llm_provider or "").strip().lower()
    if provider in CANNED_PROVIDERS:
        return None
    return ChatCompletionsProvider()


def suggest_with_fallback(
    turns: list[dict[str, str]],
    user_gender: str | None = None,
) -> tuple[list[ReplyOption], bool]:
    llm = get_llm()
    if llm is None:
        log.info("LLM_PROVIDER=%r not configured; using canned replies", settings.llm_provider)
        return canned_replies(turns), True

    key = json.dumps(
        {"turns": turns[-8:], "gender": user_gender},
        ensure_ascii=False,
        sort_keys=True,
    )
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return list(cached[1]), False

    try:
        replies = llm.suggest(turns, user_gender)
    except Exception as exc:
        log.warning("LLM failed, using canned replies: %s", exc)
        return canned_replies(turns), True

    with _CACHE_LOCK:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = (now, replies)
    return replies, False
