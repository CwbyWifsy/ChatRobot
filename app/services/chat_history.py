from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

from ..config import settings


class ChatSessionManager:
    """In-memory multi-session history tracker."""

    def __init__(self) -> None:
        self.sessions: Dict[str, Deque[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self.sessions.get(session_id, []))

    def append(self, session_id: str, user_message: str, assistant_message: str) -> None:
        history = self.sessions.setdefault(session_id, deque(maxlen=settings.max_history_turns))
        history.append({"user": user_message, "assistant": assistant_message})

    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


__all__ = ["ChatSessionManager"]
