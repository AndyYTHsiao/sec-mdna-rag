import faiss
import numpy as np
import pytest
from rank_bm25 import BM25Okapi
from src.retrieval import dense_retrieval, reciprocal_rank_fusion, sparse_retrieval
from src.utils import tokenize


@pytest.fixture
def faiss_index():
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1, 0], [0, 1], [0.8, 0.2]], dtype=np.float32))
    return index


def test_dense_retrieval_returns_highest_inner_product_first(faiss_index):
    indices, scores = dense_retrieval(
        np.array([1, 0], dtype=np.float32), faiss_index, top_k=2
    )

    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_allclose(scores, [1, 0.8])


def test_dense_candidate_filter_can_select_result_beyond_top_k(faiss_index):
    indices, _ = dense_retrieval(
        np.array([1, 0], dtype=np.float32),
        faiss_index,
        top_k=1,
        candidate_indices=np.array([1]),
    )

    np.testing.assert_array_equal(indices, [1])


def test_dense_retrieval_rejects_query_with_wrong_dimension(faiss_index):
    with pytest.raises(ValueError, match="Query dimension"):
        dense_retrieval(np.array([1, 0, 0]), faiss_index)


def test_sparse_retrieval_ranks_match_and_honors_candidates():
    bm25 = BM25Okapi([tokenize(text) for text in ("apple revenue", "cloud expense")])

    indices, _ = sparse_retrieval(
        "apple sale", bm25, top_k=2, candidate_indices=np.array([0])
    )

    np.testing.assert_array_equal(indices, [0])


def test_rrf_documents_in_both_rankings_receive_both_contributions():
    scores = reciprocal_rank_fusion(np.array([7, 8]), np.array([8, 9]), rrf_k=0)

    assert scores == {7: 1.0, 8: 1.5, 9: 0.5}


def test_rrf_rejects_negative_constant():
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion(np.array([]), np.array([]), rrf_k=-1)
