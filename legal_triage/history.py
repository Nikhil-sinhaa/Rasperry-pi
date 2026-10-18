"""Persistent chat-session history stored as local JSON files.

Sessions are kept in ~/.legal_triage_history/ — a hidden directory in the
user's home folder.  No document content leaves the device; the stored data
is the same analysis results that already live in Streamlit session_state.

File format per session  (one .json file per upload session):
{
    "session_id": "<sha1-hex>",
    "files": ["contract.pdf"],
    "created_at": "2025-10-18T15:30:00",
    "messages": [
        {"role": "user",  "content": "What is the notice period?"},
        {"role": "assistant", "question": "...", "mode": "question", "result": {...},
         "checklist": [...], "structured_fields": [...]}
    ]
}

Privacy note: this module never writes raw document bytes to disk.  The
`result` dicts stored in each assistant message contain only model-generated
text, verbatim quotes already extracted from the document, and similarity
scores — the same information displayed to the user on-screen.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

_HISTORY_DIR = Path.home() / ".legal_triage_history"


def _ensure_dir() -> Path:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return _HISTORY_DIR


def _session_path(session_id: str) -> Path:
    return _ensure_dir() / f"{session_id}.json"


def _make_session_id(filenames: list[str], created_at: str) -> str:
    key = ",".join(sorted(filenames)) + "|" + created_at
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def save_session(
    session_id: str,
    filenames: list[str],
    messages: list[dict[str, Any]],
    created_at: str | None = None,
) -> None:
    """Write or overwrite a session file."""
    path = _session_path(session_id)
    payload = {
        "session_id": session_id,
        "files": filenames,
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
        "messages": messages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(session_id: str) -> dict[str, Any] | None:
    """Return the session dict or None if not found / corrupt."""
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_sessions() -> list[dict[str, Any]]:
    """Return all saved sessions sorted newest-first.

    Each entry: {"session_id", "files", "created_at", "message_count"}
    """
    sessions: list[dict[str, Any]] = []
    try:
        dir_path = _ensure_dir()
    except OSError:
        return sessions
    for path in dir_path.glob("*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            sessions.append({
                "session_id": raw.get("session_id", path.stem),
                "files": raw.get("files", []),
                "created_at": raw.get("created_at", ""),
                "message_count": len(raw.get("messages", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a session file. Returns True if it existed and was deleted."""
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False


def make_session_id(filenames: list[str]) -> str:
    """Create a stable session ID from the list of uploaded filenames."""
    created_at = datetime.now().isoformat(timespec="seconds")
    return _make_session_id(filenames, created_at)
