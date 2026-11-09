from __future__ import annotations

from typing import Literal, Protocol

Emotion = Literal["warm", "urgent", "playful", "calm"]


class ReplyOption:
    def __init__(self, text: str, emotion: Emotion) -> None:
        self.text = text
        self.emotion = emotion

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "emotion": self.emotion}


class SuggestLLM(Protocol):
    def suggest(
        self,
        turns: list[dict[str, str]],
        user_gender: str | None = None,
        language_hint: str | None = None,
    ) -> list[ReplyOption]:
        """Return up to 3 short replies in the partner's language/mix."""
        ...
