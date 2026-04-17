"""
RAG-based explainability layer.
Answers "Why was this transaction flagged?" by:
  1. Encoding triggered features as a natural-language query
  2. Retrieving the top-k most relevant policy/rule documents via FAISS semantic search
  3. Passing retrieved context + SHAP evidence to an LLM for a concise human-readable explanation
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
import anthropic


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"

_SYSTEM_PROMPT = """You are a fraud analyst AI assistant.
Your job is to explain, in plain English, why a specific transaction was flagged as potentially fraudulent.
You are given:
- Key signals from the transaction (anomaly score, top SHAP features)
- Relevant policy rules retrieved from the company fraud rule corpus

Write a short, factual explanation (3-5 sentences) that a human fraud analyst can act on.
Be specific about which behaviours triggered concern. Do not speculate beyond the evidence."""


class RAGExplainer:
    """
    Retrieval-Augmented Generation explainer over a fraud policy corpus.
    Indexes all .txt files under the policies/ directory at construction time.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        chunk_size: int = 200,
    ):
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.embed_model = SentenceTransformer(embedding_model)
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: list[str] = []
        self._client = anthropic.Anthropic()
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        """Load policy documents, chunk them, embed, and build FAISS index."""
        policy_files = list(POLICY_DIR.glob("*.txt")) + list(POLICY_DIR.glob("*.md"))
        if not policy_files:
            logger.warning("No policy documents found in {}. RAG context will be empty.", POLICY_DIR)
            return

        raw_text = []
        for fp in policy_files:
            text = fp.read_text(encoding="utf-8")
            raw_text.append(text)
            logger.debug("Loaded policy file: {}", fp.name)

        combined = "\n\n".join(raw_text)
        # Split into sentence-level chunks with overlap
        sentences = re.split(r"(?<=[.!?])\s+", combined)
        chunks = []
        buf, buf_len = [], 0
        for sent in sentences:
            words = len(sent.split())
            if buf_len + words > self.chunk_size and buf:
                chunks.append(" ".join(buf))
                buf, buf_len = buf[-2:], sum(len(s.split()) for s in buf[-2:])
            buf.append(sent)
            buf_len += words
        if buf:
            chunks.append(" ".join(buf))

        self._chunks = chunks
        embeddings = self.embed_model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        logger.success("FAISS index built: {} chunks, {} dimensions.", len(chunks), dim)

    def save_index(self, path=None) -> Path:
        path = Path(path) if path else ARTIFACT_DIR / "faiss.index"
        if self._index:
            faiss.write_index(self._index, str(path))
            chunk_path = path.with_suffix(".chunks.json")
            chunk_path.write_text(json.dumps(self._chunks))
            logger.info("FAISS index saved → {}", path)
        return path

    def load_index(self, path=None) -> "RAGExplainer":
        path = Path(path) if path else ARTIFACT_DIR / "faiss.index"
        self._index = faiss.read_index(str(path))
        chunk_path = path.with_suffix(".chunks.json")
        self._chunks = json.loads(chunk_path.read_text())
        logger.info("FAISS index loaded ← {}", path)
        return self

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[str]:
        """Return top-k policy chunks most relevant to the query."""
        if self._index is None or self._index.ntotal == 0:
            return []
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        _, indices = self._index.search(q_emb, self.top_k)
        return [self._chunks[i] for i in indices[0] if i < len(self._chunks)]

    # ------------------------------------------------------------------
    # Explanation generation
    # ------------------------------------------------------------------

    def explain(
        self,
        transaction: dict,
        fraud_score: float,
        anomaly_score: float,
        shap_contributions: dict,
    ) -> str:
        """
        Generate a human-readable fraud explanation.

        Args:
            transaction: Raw transaction fields (amount, merchant, timestamp, etc.)
            fraud_score: Ensemble model probability [0,1]
            anomaly_score: Unsupervised anomaly score [0,1]
            shap_contributions: {feature_name: shap_value} for top features

        Returns:
            Natural-language explanation string.
        """
        query = self._build_query(transaction, fraud_score, shap_contributions)
        context_chunks = self.retrieve(query)
        context = "\n\n".join(f"[Policy excerpt]\n{c}" for c in context_chunks)

        # Format SHAP evidence
        top_features = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        shap_text = "\n".join(
            f"  - {feat}: {val:+.4f} ({'increases' if val > 0 else 'decreases'} fraud risk)"
            for feat, val in top_features
        )

        user_msg = f"""Transaction details:
{json.dumps(transaction, indent=2, default=str)}

Model scores:
  Fraud probability: {fraud_score:.4f}
  Anomaly score:     {anomaly_score:.4f}

Top SHAP feature contributions:
{shap_text}

Relevant policy context:
{context}

Please explain why this transaction was flagged."""

        response = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        explanation = response.content[0].text
        logger.debug("RAG explanation generated ({} chars).", len(explanation))
        return explanation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(transaction: dict, fraud_score: float, shap: dict) -> str:
        top = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        feature_str = ", ".join(f[0].replace("_", " ") for f, _ in top)
        amount = transaction.get("amount", "unknown")
        merchant = transaction.get("merchant_category", "unknown")
        return (
            f"Transaction amount {amount} in {merchant} category "
            f"flagged with fraud score {fraud_score:.2f}. "
            f"Key signals: {feature_str}."
        )
