"""
llm.py - the language-model boundary.

Two backends sit behind one interface:

  * MockLLM (DEFAULT, MOCK_LLM=1)
        Fully deterministic, keyless, offline. It still receives a real,
        fully-rendered prompt string - the prompt is what the transcripts
        show - but instead of sampling tokens it executes a deterministic
        policy over the retrieved context using the same local MiniLM
        encoder the retriever uses. Because it can only select sentences
        that are physically present in the supplied context, it is
        extractive by construction and cannot hallucinate.

  * RealLLM  (opt-in only, MOCK_LLM=0 plus OPENAI_API_KEY)
        Wired for completeness. Nothing in the acceptance criteria needs it.

Every graded transcript in transcripts/ was produced with MockLLM.
"""

import os
import re
from typing import List

import numpy as np

from config import mock_llm_enabled
from embeddings import cosine, embed, embed_one

# --------------------------------------------------------------------------- #
# prompt templates (rendered and shown in the transcripts)
# --------------------------------------------------------------------------- #

GROUNDED_SYSTEM_PROMPT = (
    "You are the Naukri.com employer-support agent. Answer strictly and only "
    "from the CONTEXT block below. Never use outside knowledge. If the CONTEXT "
    "does not contain the answer, reply exactly with \"I don't know\"."
)

GROUNDED_USER_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

Answer using only sentences supported by the CONTEXT."""

JUDGE_SYSTEM_PROMPT = (
    "You are a strict RAG evaluation judge. You score one (question, context, "
    "answer) triple on three independent axes, each in [0.0, 1.0]:\n"
    "  context_relevance : does the retrieved CONTEXT address the QUESTION?\n"
    "  groundedness      : is every claim in the ANSWER supported by the CONTEXT?\n"
    "  answer_relevance  : does the ANSWER actually respond to the QUESTION?\n"
    "Return only a JSON object with those three keys."
)

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}

Score context_relevance, groundedness and answer_relevance in [0.0, 1.0] and
return JSON only."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences. Shared by the mock LLM and the chunker."""
    text = re.sub(r"^#.*$", "", text, flags=re.MULTILINE)  # drop markdown headings
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


# --------------------------------------------------------------------------- #
# mock backend
# --------------------------------------------------------------------------- #

class MockLLM:
    """Deterministic, offline stand-in for a chat model."""

    name = "mock-deterministic-v1"

    #: a candidate sentence must be at least this similar to the question
    #: before the mock will put it in the answer
    SENTENCE_FLOOR = 0.22
    MAX_SENTENCES = 3

    # -- grounded generation ------------------------------------------------ #
    def generate_grounded(self, question: str, contexts: List[str]) -> str:
        """
        Extractive answer: rank every sentence in the retrieved context by
        similarity to the question and emit the best few, verbatim.
        """
        sentences: List[str] = []
        for c in contexts:
            for s in split_sentences(c):
                if s not in sentences:
                    sentences.append(s)
        if not sentences:
            return "I don't know."

        q = np.asarray(embed_one(question), dtype=np.float32)
        mat = embed(sentences)
        scored = sorted(
            ((float(np.dot(q, mat[i])), sentences[i]) for i in range(len(sentences))),
            key=lambda t: -t[0],
        )
        picked = [s for sc, s in scored[: self.MAX_SENTENCES] if sc >= self.SENTENCE_FLOOR]
        if not picked:
            return "I don't know."
        # keep the original reading order of the retrieved context
        picked.sort(key=sentences.index)
        return " ".join(picked)

    # -- LLM-as-judge ------------------------------------------------------- #
    def judge_triad(self, question: str, contexts: List[str], answer: str) -> dict:
        """
        Deterministic realisation of the RAG-triad judge. Each axis is computed
        from the same evidence a human judge would look at:

          context_relevance : best question<->retrieved-chunk similarity
          groundedness      : fraction of answer sentences that are supported by
                              some context sentence (similarity >= 0.60), which
                              is the axis a hallucination would break
          answer_relevance  : question<->answer similarity

        Refusals are scored by an explicit convention rather than by similarity,
        because a refusal is a different kind of object from an answer. Both
        refusal forms - the groundedness fallback and the prompt-injection
        refusal - assert no claim about hiring policy, so they are vacuously
        grounded (1.0), and they are scored relevant exactly to the extent that
        the retrieved context had nothing to offer (1 - context_relevance). A
        refusal issued when good context WAS available therefore scores badly on
        answer relevance, which is the behaviour that would catch an
        over-refusing agent.
        """
        ctx_text = [c for c in contexts if c.strip()]
        refusal = is_refusal(answer)

        # --- context relevance ---
        if ctx_text:
            q = np.asarray(embed_one(question), dtype=np.float32)
            cmat = embed(ctx_text)
            context_relevance = float(np.max(cmat @ q))
        else:
            context_relevance = 0.0
        context_relevance = _clip01(context_relevance)

        # --- groundedness ---
        if refusal:
            # a correct refusal makes no unsupported claim
            groundedness = 1.0
        else:
            ans_sents = split_sentences(answer)
            ctx_sents: List[str] = []
            for c in ctx_text:
                ctx_sents.extend(split_sentences(c))
            if not ans_sents or not ctx_sents:
                groundedness = 0.0
            else:
                amat = embed(ans_sents)
                cmat2 = embed(ctx_sents)
                sims = amat @ cmat2.T                      # (n_ans, n_ctx)
                best = sims.max(axis=1)
                groundedness = float(np.mean(best >= 0.60))
        groundedness = _clip01(groundedness)

        # --- answer relevance ---
        if refusal:
            # a refusal is the *right* answer when nothing relevant was found,
            # and the wrong one otherwise; score it by how little the context
            # actually had to offer
            answer_relevance = _clip01(1.0 - context_relevance)
        else:
            answer_relevance = _clip01(
                cosine(embed_one(question), embed_one(answer))
            )

        return {
            "context_relevance": round(context_relevance, 4),
            "groundedness": round(groundedness, 4),
            "answer_relevance": round(answer_relevance, 4),
        }


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


#: Both ways the agent can decline: the groundedness fallback and the
#: prompt-injection refusal. Either one asserts no factual claim about the
#: hiring policies, so the judge must treat them the same way.
REFUSAL_PREFIXES = ("i don't know", "i can't act on that")


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith(REFUSAL_PREFIXES)


# --------------------------------------------------------------------------- #
# optional real backend
# --------------------------------------------------------------------------- #

class RealLLM:
    """Opt-in OpenAI-compatible backend. Never required by this project."""

    name = "openai-real"

    def __init__(self):
        from openai import OpenAI  # imported lazily so MOCK mode needs nothing

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = os.getenv("REAL_LLM_MODEL", "gpt-4o-mini")

    def _chat(self, system: str, user: str) -> str:
        r = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return r.choices[0].message.content.strip()

    def generate_grounded(self, question: str, contexts: List[str]) -> str:
        return self._chat(
            GROUNDED_SYSTEM_PROMPT,
            GROUNDED_USER_TEMPLATE.format(
                context="\n\n".join(contexts), question=question
            ),
        )

    def judge_triad(self, question: str, contexts: List[str], answer: str) -> dict:
        import json

        raw = self._chat(
            JUDGE_SYSTEM_PROMPT,
            JUDGE_USER_TEMPLATE.format(
                question=question, context="\n\n".join(contexts), answer=answer
            ),
        )
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
        d = json.loads(raw)
        return {k: _clip01(float(d[k])) for k in
                ("context_relevance", "groundedness", "answer_relevance")}


_backend = None


def get_llm():
    """Return the active backend - MockLLM unless the operator opted out."""
    global _backend
    if _backend is None:
        _backend = MockLLM() if mock_llm_enabled() else RealLLM()
    return _backend
