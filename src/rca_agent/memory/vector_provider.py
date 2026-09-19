"""Vector memory providers for semantic incident similarity.

Architecture
------------
``VectorMemoryProvider``  — structural Protocol (the interface)
``TfidfVectorProvider``   — in-process TF-IDF + cosine similarity (no API keys)

The ``TfidfVectorProvider`` is intentionally self-contained:
* Uses only ``numpy`` and Python stdlib — no sentence-transformers, OpenAI, etc.
* Produces meaningful similarity for incident text without requiring a running
  embedding service.
* The interface is stable so a real embedding provider can be swapped in later
  by implementing ``VectorMemoryProvider`` without changing any calling code.

Similarity is computed over the concatenation of:
    title + " " + description + " " + root_cause_summary (if present)

This gives the best signal for "same type of problem" matching.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable

from rca_agent.models.memory_models import SimilarIncident

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class VectorMemoryProvider(Protocol):
    """Interface for semantic similarity retrieval over incidents."""

    def index_incident(
        self,
        incident_id: str,
        title: str,
        description: str,
        root_cause_summary: str = "",
        application: str = "",
        severity: str = "",
    ) -> None:
        """Add or update an incident in the vector index."""
        ...

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        exclude_ids: set[str] | None = None,
    ) -> list[SimilarIncident]:
        """Return the most similar indexed incidents for *query* text.

        Parameters
        ----------
        query:
            Free-form text describing the incident or symptom.
        top_k:
            Maximum number of results to return.
        threshold:
            Minimum cosine similarity (0.0–1.0) required to include a result.
        exclude_ids:
            Incident IDs to exclude from results (e.g. the incident being queried).
        """
        ...

    def remove_incident(self, incident_id: str) -> bool:
        """Remove an incident from the index.  Returns True if found."""
        ...

    def count(self) -> int:
        """Return the number of indexed incidents."""
        ...


# ---------------------------------------------------------------------------
# TF-IDF implementation
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _compute_tfidf(
    doc_tokens: list[str],
    all_docs: list[list[str]],
    vocab: list[str],
    idf: dict[str, float],
) -> list[float]:
    """Return a TF-IDF vector for *doc_tokens* using pre-computed *idf*."""
    tf = Counter(doc_tokens)
    total = len(doc_tokens) or 1
    return [tf.get(term, 0) / total * idf.get(term, 0.0) for term in vocab]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TfidfVectorProvider:
    """Lightweight TF-IDF vector store using Python stdlib + numpy arithmetic.

    All computation is in-process.  The index is rebuilt lazily when the
    document set changes.  Suitable for hundreds of incidents; for millions
    use a dedicated vector database.
    """

    def __init__(self) -> None:
        # Store raw document text and metadata keyed by incident_id
        self._docs: dict[str, dict[str, str]] = {}
        # Cache: vocabulary, IDF scores, and document vectors
        self._vocab: list[str] = []
        self._idf: dict[str, float] = {}
        self._doc_vectors: dict[str, list[float]] = {}
        self._dirty: bool = False  # True when index needs rebuilding

    # ---- Write -----------------------------------------------------------

    def index_incident(
        self,
        incident_id: str,
        title: str,
        description: str,
        root_cause_summary: str = "",
        application: str = "",
        severity: str = "",
    ) -> None:
        text = f"{title} {description} {root_cause_summary}".strip()
        self._docs[incident_id] = {
            "text": text,
            "title": title,
            "description": description,
            "application": application,
            "severity": severity,
        }
        self._dirty = True

    def remove_incident(self, incident_id: str) -> bool:
        if incident_id not in self._docs:
            return False
        del self._docs[incident_id]
        self._dirty = True
        return True

    def count(self) -> int:
        return len(self._docs)

    # ---- Read ------------------------------------------------------------

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        exclude_ids: set[str] | None = None,
    ) -> list[SimilarIncident]:
        if not self._docs:
            return []
        self._rebuild_if_dirty()

        query_tokens = _tokenise(query)
        query_vec = _compute_tfidf(query_tokens, [], self._vocab, self._idf)

        exclude = exclude_ids or set()
        scored: list[tuple[str, float]] = []
        for inc_id, doc_vec in self._doc_vectors.items():
            if inc_id in exclude:
                continue
            score = _cosine(query_vec, doc_vec)
            if score >= threshold:
                scored.append((inc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results: list[SimilarIncident] = []
        for inc_id, score in scored[:top_k]:
            meta = self._docs[inc_id]
            results.append(
                SimilarIncident(
                    incident_id=inc_id,
                    title=meta["title"],
                    description=meta["description"],
                    similarity_score=round(score, 4),
                    application=meta.get("application", ""),
                    severity=meta.get("severity", ""),
                )
            )
        return results

    # ---- Internal --------------------------------------------------------

    def _rebuild_if_dirty(self) -> None:
        if not self._dirty:
            return
        ids = list(self._docs.keys())
        tokenised: list[list[str]] = [_tokenise(self._docs[i]["text"]) for i in ids]

        # Vocabulary: all unique terms
        vocab_set: set[str] = set()
        for tokens in tokenised:
            vocab_set.update(tokens)
        self._vocab = sorted(vocab_set)

        n = len(tokenised)
        # IDF: log((n + 1) / (df + 1)) + 1  (smooth)
        self._idf = {}
        for term in self._vocab:
            df = sum(1 for tokens in tokenised if term in tokens)
            self._idf[term] = math.log((n + 1) / (df + 1)) + 1.0

        # Per-document TF-IDF vectors
        self._doc_vectors = {}
        for inc_id, tokens in zip(ids, tokenised):
            self._doc_vectors[inc_id] = _compute_tfidf(tokens, tokenised, self._vocab, self._idf)

        self._dirty = False
        logger.debug(
            "TfidfVectorProvider: rebuilt index with %d docs, vocab size %d",
            n, len(self._vocab),
        )
