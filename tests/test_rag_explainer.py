"""Tests for RAGExplainer — mocked to avoid Anthropic API calls."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

# Stub heavy optional dependencies before any imports of the module
_mock_faiss_index = MagicMock()
_mock_faiss_index.search.return_value = (np.zeros((1, 3)), np.zeros((1, 3), dtype=np.int64))
_mock_faiss_index.add = MagicMock()

_faiss_stub = ModuleType("faiss")
_faiss_stub.IndexFlatIP = MagicMock(return_value=_mock_faiss_index)
sys.modules.setdefault("faiss", _faiss_stub)

_mock_encoder = MagicMock()
_mock_encoder.get_sentence_embedding_dimension.return_value = 384
_mock_encoder.encode.return_value = np.zeros((1, 384), dtype=np.float32)

_st_stub = ModuleType("sentence_transformers")
_st_stub.SentenceTransformer = MagicMock(return_value=_mock_encoder)
sys.modules.setdefault("sentence_transformers", _st_stub)

_mock_content = MagicMock()
_mock_content.text = "This transaction was flagged due to high fraud score."
_mock_response = MagicMock()
_mock_response.content = [_mock_content]
_mock_anthropic_client = MagicMock()
_mock_anthropic_client.messages.create.return_value = _mock_response

_anthropic_stub = ModuleType("anthropic")
_anthropic_stub.Anthropic = MagicMock(return_value=_mock_anthropic_client)
sys.modules.setdefault("anthropic", _anthropic_stub)


@pytest.fixture()
def sample_transaction() -> dict:
    return {
        "user_id": 12345,
        "amount": 9999.99,
        "merchant_category": "crypto",
        "payment_method": "wire",
        "device_type": "mobile",
        "channel": "online",
        "account_age_days": 7,
        "credit_utilization": 0.95,
        "prior_fraud_count": 1,
    }


@pytest.fixture(scope="module")
def explainer():
    if "models.rag.rag_explainer" in sys.modules:
        del sys.modules["models.rag.rag_explainer"]
    from models.rag.rag_explainer import RAGExplainer
    return RAGExplainer()


class TestRAGExplainerInit:
    def test_explainer_creates_instance(self, explainer):
        assert explainer is not None

    def test_top_k_default(self, explainer):
        assert explainer.top_k == 3

    def test_embed_model_exists(self, explainer):
        assert hasattr(explainer, "embed_model")

    def test_chunks_populated(self, explainer):
        assert isinstance(explainer._chunks, list)


class TestRAGExplainerExplain:
    def test_explain_returns_string(self, explainer, sample_transaction):
        result = explainer.explain(
            transaction=sample_transaction,
            fraud_score=0.92,
            anomaly_score=0.85,
            shap_contributions={"amount_zscore": 0.4},
        )
        assert isinstance(result, str)

    def test_explain_no_shap_still_works(self, explainer, sample_transaction):
        result = explainer.explain(
            transaction=sample_transaction,
            fraud_score=0.7,
            anomaly_score=0.5,
            shap_contributions={},
        )
        assert isinstance(result, str)

    @pytest.mark.parametrize("fraud_score,anomaly_score", [
        (0.1, 0.05),
        (0.5, 0.5),
        (0.95, 0.9),
    ])
    def test_explain_various_scores(self, explainer, sample_transaction, fraud_score, anomaly_score):
        result = explainer.explain(
            transaction=sample_transaction,
            fraud_score=fraud_score,
            anomaly_score=anomaly_score,
            shap_contributions={},
        )
        assert isinstance(result, str)


class TestRAGExplainerBuildQuery:
    def test_build_query_contains_amount(self, sample_transaction):
        if "models.rag.rag_explainer" in sys.modules:
            del sys.modules["models.rag.rag_explainer"]
        from models.rag.rag_explainer import RAGExplainer
        query = RAGExplainer._build_query(sample_transaction, 0.9, {"amount_zscore": 0.5})
        assert "9999.99" in query

    def test_build_query_contains_score(self, sample_transaction):
        if "models.rag.rag_explainer" in sys.modules:
            del sys.modules["models.rag.rag_explainer"]
        from models.rag.rag_explainer import RAGExplainer
        query = RAGExplainer._build_query(sample_transaction, 0.92, {})
        assert "0.92" in query
