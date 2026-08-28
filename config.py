"""
config.py - shared paths and constants for the Naukri.com support agent.

Everything in this project runs with MOCK_LLM=1 (the default), which needs
zero API keys and zero network access at run time.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

KB_DIR = ROOT / "knowledge_base"
CHROMA_DIR = ROOT / "chroma_store"
TRANSCRIPT_DIR = ROOT / "transcripts"
LOG_DIR = ROOT / "logs"

MEMORY_FILE = ROOT / "conversation_memory.json"
CHECKPOINT_DB = ROOT / "checkpoints.sqlite"
REQUEST_LOG = LOG_DIR / "requests.jsonl"

# --- embedding / retrieval ---------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_FIXED = "naukri_kb_fixed_overlap"
COLLECTION_SENTENCE = "naukri_kb_sentence"

# fixed-size chunking parameters (characters)
FIXED_CHUNK_SIZE = 420
FIXED_CHUNK_OVERLAP = 90

# sentence-based chunking parameters
SENTENCE_GROUP_SIZE = 2      # sentences per chunk
SENTENCE_GROUP_STRIDE = 1    # sliding window -> 1 sentence of overlap

TOP_K = 3

# --- grounded-generation "I don't know" threshold ----------------------------
# NOT a tutorial preset. Measured by calibrate_threshold.py against this exact
# knowledge base and encoder (6 in-scope + 3 out-of-scope queries, both
# collections). Raw numbers are reproduced in README.md and in
# transcripts/04_threshold_calibration.txt:
#
#   sentence-based collection : in-scope 0.6930..0.8142 | out-of-scope 0.1451..0.1932
#                               gap 0.4998 -> midpoint 0.4431
#   fixed-size collection     : in-scope 0.6230..0.7816 | out-of-scope 0.1424..0.1979
#                               gap 0.4251 -> midpoint 0.4104
#
# The LOWER of the two midpoints is used so one threshold is safe for either
# collection: it sits above every measured out-of-scope score and below every
# measured in-scope score in both indexes.
GROUNDEDNESS_THRESHOLD = 0.4104

FALLBACK_ANSWER = (
    "I don't know. I could not find anything in the Naukri.com employer-support "
    "knowledge base that answers this question, so I will not guess. Please "
    "rephrase the question or contact the employer-support desk."
)

# --- LLM mode ----------------------------------------------------------------
def mock_llm_enabled() -> bool:
    """True unless the operator explicitly opts into a real LLM backend."""
    return os.getenv("MOCK_LLM", "1").strip().lower() not in ("0", "false", "no")


for _d in (CHROMA_DIR, TRANSCRIPT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
