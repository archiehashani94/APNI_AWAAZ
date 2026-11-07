from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass
class Session:
    id: str
    consent: bool = False
    speaker_embedding: dict[str, Any] | None = None
    clone_ready: bool = False
    using_stock_voice: bool = False
    turns: list[dict[str, str]] = field(default_factory=list)
    favourite_audio: dict[str, bytes] = field(default_factory=dict)


class SessionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session(id=uuid4().hex)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def require(self, session_id: str) -> Session:
        session = self.get(session_id)
        if not session:
            session = Session(id=session_id)
            with self._lock:
                self._sessions[session_id] = session
        return session


store = SessionStore()
