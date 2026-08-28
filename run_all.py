"""
run_all.py - regenerate every transcript in transcripts/ from a clean state.

    python run_all.py

Each step runs as its own subprocess with MOCK_LLM=1, and its combined
stdout/stderr is written to transcripts/<name>.txt as well as echoed here.
Noise from the embedding model loader (HuggingFace download bars, symlink
warnings) is filtered out so the transcripts stay readable.

The step order matters in one place: step 11/12 exercises POST /add-document,
which indexes an extra document into the live ChromaDB collections. The index
is therefore rebuilt immediately before the RAG-triad evaluation so that
evaluation always scores the same 12-document knowledge base.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from config import TRANSCRIPT_DIR

ROOT = Path(__file__).resolve().parent

NOISE = re.compile(
    r"(Loading weights|it/s\]|B/s\]|huggingface_hub|HF_TOKEN|symlink|"
    r"warnings\.warn|Developer Mode|docs\.microsoft\.com|"
    r"To support symlinks|^\s*$\Z)",
    re.IGNORECASE,
)

STEPS = [
    ("01_dataset",              ["dataset.py"],
     "Part 1 / Task 1 - dataset design and validation"),
    ("02_chunking_and_index",   ["rag_core.py", "index"],
     "Part 1 / Tasks 2-3 - knowledge base, two chunking strategies, two collections"),
    ("03_threshold_calibration", ["calibrate_threshold.py"],
     "Part 1 / Task 4 - empirical 'I don't know' threshold calibration"),
    ("04_grounded_generation",  ["rag_core.py", "demo"],
     "Part 1 / Task 4 - grounded generation, 5 in-scope + 1 out-of-scope"),
    ("05_chunking_comparison",  ["eval_chunking.py"],
     "Part 1 / Task 5 - Precision@3 / Recall@3 for both collections"),
    ("06_tool_escalation",      ["tools.py"],
     "Part 2 / Task 6 - check_job_application_status + escalation score"),
    ("07_agent_routing",        ["agent.py", "demo"],
     "Part 2 / Task 7 - LangGraph routing, both branches firing"),
    ("08_memory",               ["agent.py", "memory"],
     "Part 2 / Task 8 - multi-turn memory + fresh-conversation reset"),
    ("09_output_schema",        ["schemas.py"],
     "Part 2 / Task 9 - structured output JSON Schema"),
    ("10a_guardrails_unit",     ["guardrails.py"],
     "Part 2 / Task 10 - all three guardrails firing"),
    ("10b_guardrails_in_graph", ["agent.py", "guardrails"],
     "Part 2 / Task 10 - the same guardrails firing inside the graph"),
    ("11_12_fastapi_logging",   ["app.py", "demo"],
     "Part 3 / Tasks 11-12 - FastAPI endpoints + JSON-Lines logging"),
    ("__reindex__",             ["rag_core.py", "index"],
     "(housekeeping: restore the 12-document index after /add-document)"),
    ("13_rag_triad",            ["eval_rag_triad.py"],
     "Part 3 / Task 13 - RAG triad over 15 queries"),
    ("14_mcp_roundtrip",        ["mcp_client.py", "--spawn-server"],
     "Part 4 / Task 14 - MCP client/server round trip"),
    ("15_checkpointing",        ["checkpoint_demo.py"],
     "Part 4 / Task 15 - SQLite checkpointing across an interruption"),
    ("16_resilience",           ["resilience_demo.py"],
     "Part 4 / Task 16 - retries, per-node timeout, global timeout"),
]


def clean(text: str) -> str:
    out = []
    for line in text.splitlines():
        if NOISE.search(line):
            continue
        out.append(line.rstrip())
    # collapse runs of blank lines
    result, blank = [], 0
    for line in out:
        if not line.strip():
            blank += 1
            if blank > 2:
                continue
        else:
            blank = 0
        result.append(line)
    return "\n".join(result).strip() + "\n"


def main() -> int:
    # optional filter: `python run_all.py 13` regenerates only matching steps
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")]
    steps = ([s for s in STEPS if any(w in s[0] for w in wanted)]
             if wanted else STEPS)
    if not steps:
        print(f"no steps matched {wanted}; available: "
              f"{[s[0] for s in STEPS if s[0] != '__reindex__']}")
        return 1

    env = dict(os.environ)
    env["MOCK_LLM"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["ANONYMIZED_TELEMETRY"] = "False"

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    failures = []
    print("=" * 78)
    print("REGENERATING ALL TRANSCRIPTS (MOCK_LLM=1, no API keys)")
    print("=" * 78)

    for name, argv, description in steps:
        label = description if name == "__reindex__" else f"{name}  -  {description}"
        print(f"\n>>> {label}")
        sys.stdout.flush()

        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable] + argv,
            cwd=str(ROOT), env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        elapsed = time.perf_counter() - t0
        body = clean((proc.stdout or "") + (proc.stderr or ""))

        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"    {status} in {elapsed:.1f}s")
        if proc.returncode != 0:
            failures.append(name)
            print("    --- tail of output ---")
            for line in body.splitlines()[-15:]:
                print(f"    {line}")

        if name == "__reindex__":
            continue

        header = (
            f"{'=' * 78}\n"
            f"TRANSCRIPT: {name}\n"
            f"{description}\n"
            f"command   : python {' '.join(argv)}\n"
            f"MOCK_LLM  : 1 (no API key, no network)\n"
            f"exit code : {proc.returncode}\n"
            f"duration  : {elapsed:.1f}s\n"
            f"{'=' * 78}\n\n"
        )
        (TRANSCRIPT_DIR / f"{name}.txt").write_text(header + body, encoding="utf-8")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED STEPS: {failures}")
    else:
        print("ALL STEPS PASSED")
    print(f"transcripts written to {TRANSCRIPT_DIR}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
