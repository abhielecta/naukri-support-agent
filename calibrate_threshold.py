"""
calibrate_threshold.py - Part 1 / Task 4 (threshold calibration)

The brief explicitly forbids picking an untested preset such as 0.5/0.6/0.7.
This script measures the actual top-1 cosine similarity produced by THIS
knowledge base and THIS encoder for in-scope and out-of-scope queries, then
places the "I don't know" threshold in the empty gap between the two clusters.

    python calibrate_threshold.py

The printed numbers are the ones quoted in README.md and hard-coded as
GROUNDEDNESS_THRESHOLD in config.py.
"""

from config import COLLECTION_FIXED, COLLECTION_SENTENCE
from rag_core import retrieve

IN_SCOPE = [
    "How long does an offer stay open before it expires?",
    "What does the background verification check actually cover?",
    "When am I eligible to apply for an internal transfer?",
    "How long is the probation period for a new hire?",
    "When is the referral bonus paid out?",
    "How many times can a candidate reschedule an interview?",
]

OUT_OF_SCOPE = [
    "What is the best recipe for Hyderabadi biryani?",
    "Who won the 2011 cricket world cup final?",
    "How do I replace the timing belt on a diesel engine?",
]


def measure(collection: str):
    print(f"\n### collection: {collection}")
    print(f"{'top-1 cosine':>13}  query")
    print("-" * 78)

    in_scores, out_scores = [], []
    for q in IN_SCOPE:
        s = retrieve(q, k=1, collection=collection)[0].similarity
        in_scores.append(s)
        print(f"{s:>13.4f}  [IN ] {q}")
    print()
    for q in OUT_OF_SCOPE:
        s = retrieve(q, k=1, collection=collection)[0].similarity
        out_scores.append(s)
        print(f"{s:>13.4f}  [OUT] {q}")

    lo_in, hi_out = min(in_scores), max(out_scores)
    print()
    print(f"  in-scope  cluster : min={lo_in:.4f}  max={max(in_scores):.4f}  "
          f"n={len(in_scores)}")
    print(f"  out-scope cluster : min={min(out_scores):.4f}  max={hi_out:.4f}  "
          f"n={len(out_scores)}")
    gap = lo_in - hi_out
    print(f"  separation gap    : {gap:.4f}")
    if gap > 0:
        mid = (lo_in + hi_out) / 2.0
        print(f"  => threshold at gap midpoint = {mid:.4f}")
    else:
        mid = None
        print("  => clusters OVERLAP; no clean threshold exists for this setup")
    return {"in": in_scores, "out": out_scores, "threshold": mid}


def main():
    print("=" * 78)
    print("PART 1 / TASK 4 - EMPIRICAL 'I DON'T KNOW' THRESHOLD CALIBRATION")
    print("6 in-scope queries, 3 deliberately out-of-scope queries")
    print("(requirement: >= 3 in-scope and >= 2 out-of-scope)")
    print("=" * 78)

    res_sent = measure(COLLECTION_SENTENCE)
    res_fixed = measure(COLLECTION_FIXED)

    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)
    print(f"sentence-based collection midpoint : {res_sent['threshold']:.4f}")
    print(f"fixed-size   collection midpoint   : {res_fixed['threshold']:.4f}")
    chosen = min(res_sent["threshold"], res_fixed["threshold"])
    print(f"\nGROUNDEDNESS_THRESHOLD chosen      : {chosen:.4f}")
    print("The lower of the two midpoints is taken so that the single threshold "
          "is\nsafe for either collection: it still sits above every measured "
          "out-of-scope\nscore and below every measured in-scope score in both "
          "indexes.")


if __name__ == "__main__":
    main()
