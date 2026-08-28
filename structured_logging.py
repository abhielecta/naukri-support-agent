"""
structured_logging.py - Part 3 / Task 12

One JSON-Lines entry per request, appended to logs/requests.jsonl, carrying a
trace id and timing information.

PII RULE (the part the brief calls out explicitly)
--------------------------------------------------
The request text written to disk is passed through the SAME
guardrails.mask_pii() that the model sees. A fixed-format phone number
therefore never reaches the log file in the clear. log_request() masks
defensively at the point of writing even when the caller already masked,
so there is no code path that can leak a raw phone number to disk.

Note the module is deliberately NOT called logging.py - that would shadow the
standard-library module.
"""

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from config import REQUEST_LOG
from guardrails import mask_pii

_write_lock = threading.Lock()


def new_trace_id() -> str:
    return f"trace-{uuid.uuid4().hex[:12]}"


class RequestTimer:
    """Context manager that measures wall-clock latency in milliseconds."""

    def __init__(self):
        self.started_at = None
        self.elapsed_ms = None

    def __enter__(self):
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = round((time.perf_counter() - self.started_at) * 1000, 2)
        return False


def log_request(
    *,
    trace_id: str,
    endpoint: str,
    request_text: str,
    status_code: int,
    latency_ms: float,
    route: Optional[str] = None,
    conversation_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Append exactly one JSON-Lines entry and return the entry that was written.

    request_text is masked here, unconditionally, before it is serialised.
    """
    safe_text, pii_masked, _raw = mask_pii(request_text or "")

    entry: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "trace_id": trace_id,
        "endpoint": endpoint,
        "conversation_id": conversation_id,
        "request_text": safe_text,          # masked - never the raw value
        "pii_masked": pii_masked,
        "route": route,
        "status_code": status_code,
        "latency_ms": latency_ms,
    }
    if extra:
        # defensive: mask any string that a caller tries to attach
        clean = {}
        for k, v in extra.items():
            clean[k] = mask_pii(v)[0] if isinstance(v, str) else v
        entry["extra"] = clean

    REQUEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _write_lock:
        with open(REQUEST_LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return entry


def read_log(limit: int = 50):
    """Read back the last `limit` JSON-Lines entries."""
    if not REQUEST_LOG.exists():
        return []
    lines = REQUEST_LOG.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines[-limit:] if l.strip()]
