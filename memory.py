"""Small, per-browser-session conversation memory for Streamlit."""

from __future__ import annotations

from typing import Any

import streamlit as st


def initialize() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def messages() -> list[dict[str, str]]:
    initialize()
    return list(st.session_state.messages)


def add(role: str, content: str) -> None:
    initialize()
    st.session_state.messages.append({"role": role, "content": content})


def recent(limit: int = 6) -> list[dict[str, str]]:
    return messages()[-limit:]


def as_prompt(limit: int = 6) -> str:
    lines = []
    for message in recent(limit):
        lines.append(f"{message['role'].title()}: {message['content']}")
    return "\n".join(lines) or "(No earlier conversation.)"
