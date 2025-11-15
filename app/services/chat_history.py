from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from ..config import settings


@dataclass
class SessionState:
    history: Deque[Dict[str, str]] = field(
        default_factory=lambda: deque(maxlen=settings.max_history_turns)
    )
    collection: Optional[str] = None


class ChatSessionManager:
    """In-memory multi-session history tracker with per-session collection preference."""

    def __init__(self) -> None:
        self.sessions: Dict[str, SessionState] = {}

    def _get_or_create_state(self, session_id: str) -> SessionState:
        state = self.sessions.get(session_id)
        if state is None:
            state = SessionState()
            self.sessions[session_id] = state
        return state

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        state = self.sessions.get(session_id)
        if not state:
            return []
        return list(state.history)

    def get_collection(self, session_id: str) -> Optional[str]:
        state = self.sessions.get(session_id)
        if not state:
            return None
        return state.collection

    def set_collection(self, session_id: str, collection: Optional[str]) -> None:
        state = self._get_or_create_state(session_id)
        if collection and state.collection and state.collection != collection:
            state.history.clear()
        state.collection = collection

    def append(self, session_id: str, user_message: str, assistant_message: str) -> None:
        state = self._get_or_create_state(session_id)
        state.history.append({"user": user_message, "assistant": assistant_message})

    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


__all__ = ["ChatSessionManager"]
