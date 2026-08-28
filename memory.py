"""
memory.py - Part 2 / Task 8

Conversation history persisted to a JSON file on disk, keyed by conversation id
(the same string used as the LangGraph thread id in Part 4). The file survives
process exit, so a second `python` process resuming the same conversation id
sees the earlier turns.

File layout (conversation_memory.json):

    {
      "conv-alpha": {
        "conversation_id": "conv-alpha",
        "created_at": "...",
        "turns": [
          {"turn_index": 1, "role": "user",  "text": "...", "ts": "..."},
          {"turn_index": 1, "role": "agent", "text": "...", "route": "rag", ...}
        ],
        "facts": {"last_record_id": "NAU-1003"}
      }
    }

`facts` is the slot that makes multi-turn behaviour observable: when turn 1 asks
about NAU-1003 and turn 2 says "and what is the escalation score for it?", the
agent resolves "it" from facts.last_record_id.
"""

import json
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import MEMORY_FILE

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_all() -> Dict:
    if not MEMORY_FILE.exists():
        return {}
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_all(data: Dict) -> None:
    MEMORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_conversation(conversation_id: str) -> Dict:
    """Return the stored conversation, or a fresh empty one."""
    with _lock:
        data = _load_all()
    return data.get(conversation_id) or {
        "conversation_id": conversation_id,
        "created_at": _now(),
        "turns": [],
        "facts": {},
    }


def append_turn(conversation_id: str, turn_index: int, role: str, text: str,
                **extra) -> Dict:
    """Append one turn and flush the whole store to disk."""
    with _lock:
        data = _load_all()
        conv = data.get(conversation_id) or {
            "conversation_id": conversation_id,
            "created_at": _now(),
            "turns": [],
            "facts": {},
        }
        entry = {"turn_index": turn_index, "role": role, "text": text,
                 "ts": _now()}
        entry.update(extra)
        conv["turns"].append(entry)
        data[conversation_id] = conv
        _save_all(data)
    return conv


def remember_fact(conversation_id: str, key: str, value) -> None:
    with _lock:
        data = _load_all()
        conv = data.setdefault(conversation_id, {
            "conversation_id": conversation_id,
            "created_at": _now(),
            "turns": [],
            "facts": {},
        })
        conv.setdefault("facts", {})[key] = value
        _save_all(data)


def get_fact(conversation_id: str, key: str, default=None):
    return load_conversation(conversation_id).get("facts", {}).get(key, default)


def turn_count(conversation_id: str) -> int:
    """Number of USER turns already recorded for this conversation."""
    conv = load_conversation(conversation_id)
    return sum(1 for t in conv["turns"] if t["role"] == "user")


def history_text(conversation_id: str, last_n: int = 6) -> str:
    """Compact rendering of recent history, for display in transcripts."""
    conv = load_conversation(conversation_id)
    lines = []
    for t in conv["turns"][-last_n:]:
        lines.append(f"  [{t['turn_index']}] {t['role']:<5}: {t['text'][:100]}")
    return "\n".join(lines) if lines else "  <no prior turns>"


def reset_conversation(conversation_id: str) -> None:
    """Drop one conversation from the store."""
    with _lock:
        data = _load_all()
        data.pop(conversation_id, None)
        _save_all(data)


def list_conversations() -> List[str]:
    return sorted(_load_all().keys())


def reset_all() -> None:
    with _lock:
        _save_all({})


def snapshot(conversation_id: str) -> Optional[Dict]:
    """Raw stored dict, or None if this conversation has never been seen."""
    with _lock:
        return _load_all().get(conversation_id)
