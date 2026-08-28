"""
schemas.py - Part 2 / Task 9

Every response the agent emits must validate against AGENT_RESPONSE_SCHEMA
before it leaves the graph. Validation is enforced in code by
validate_agent_response(), which the final node of the LangGraph graph calls on
its own output; a response that does not conform raises and is replaced by a
schema-conformant error envelope rather than being returned malformed.

The schema is plain JSON Schema (draft 2020-12) so it is portable outside
Python; it is checked with the `jsonschema` library.
"""

from typing import Tuple

from jsonschema import Draft202012Validator, ValidationError

AGENT_RESPONSE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "NaukriSupportAgentResponse",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "trace_id",
        "route",
        "answer",
        "grounded",
        "sources",
        "guardrails",
        "escalation",
        "turn_index",
    ],
    "properties": {
        "schema_version": {"const": "1.0"},
        "trace_id": {"type": "string", "minLength": 8},
        "route": {
            "description": "which branch of the conditional edge handled this turn",
            "enum": ["rag", "record_lookup", "blocked"],
        },
        "answer": {"type": "string", "minLength": 1},
        "grounded": {
            "type": "boolean",
            "description": "False when the output-side groundedness guardrail "
                           "refused, or when the record was not found",
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "KB document titles, or the record id for a lookup",
        },
        "guardrails": {
            "type": "object",
            "additionalProperties": False,
            "required": ["pii_masked", "injection_detected",
                         "groundedness_refused", "masked_query"],
            "properties": {
                "pii_masked": {"type": "boolean"},
                "injection_detected": {"type": "boolean"},
                "groundedness_refused": {"type": "boolean"},
                "masked_query": {"type": "string"},
            },
        },
        "escalation": {
            "description": "populated only on the record_lookup route",
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["record_id", "escalation_score", "escalate"],
            "properties": {
                "record_id": {"type": "string"},
                "escalation_score": {"type": "number", "minimum": 0, "maximum": 1},
                "escalate": {"type": "boolean"},
            },
        },
        "turn_index": {"type": "integer", "minimum": 1},
        "retrieval": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "properties": {
                "top_similarity": {"type": "number"},
                "threshold": {"type": "number"},
                "doc_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

_VALIDATOR = Draft202012Validator(AGENT_RESPONSE_SCHEMA)


def validate_agent_response(payload: dict) -> Tuple[bool, str]:
    """Return (is_valid, message). Never raises."""
    try:
        _VALIDATOR.validate(payload)
        return True, "valid"
    except ValidationError as exc:
        path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        return False, f"schema violation at {path}: {exc.message}"


def assert_valid(payload: dict) -> dict:
    """Validate or raise. Used where a malformed response must not escape."""
    ok, msg = validate_agent_response(payload)
    if not ok:
        raise ValueError(msg)
    return payload


if __name__ == "__main__":
    import json

    print("=" * 78)
    print("PART 2 / TASK 9 - STRUCTURED OUTPUT SCHEMA")
    print("=" * 78)
    print(json.dumps(AGENT_RESPONSE_SCHEMA, indent=2))
