"""Small, bounded per-browser-session conversation memory."""

from __future__ import annotations

from collections import defaultdict

_sessions: dict[str, list[dict[str, str]]] = defaultdict(list)


def add(session_id: str, role: str, content: str) -> None:
    history = _sessions[session_id]
    history.append({"role": role, "content": content})
    del history[:-12]


def as_prompt(session_id: str, limit: int = 6) -> str:
    lines = []
    for message in _sessions[session_id][-limit:]:
        lines.append(f"{message['role'].title()}: {message['content']}")
    return "\n".join(lines) or "(No earlier conversation.)"
