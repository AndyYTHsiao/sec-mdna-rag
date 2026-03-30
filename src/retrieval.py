import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import Tuple

# ---------------------------
# Dense Retrieval
# ---------------------------


def dense_retrieval(
    query_emb: np.ndarray,
    index: faiss.Index,
    *,
    top_k: int = 10,
    nprobe: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dense retrieval using FAISS.

    Args:
        query_emb (np.ndarray): The embedded query.
        index (faiss.Index): Faiss index object.
        top_k (int): The number of retrieved docs.
        nprobe (int | None): Number of cells visited to perform a search.

    Returns:
        indices (top_k,)
        scores  (top_k,)
    """
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)

    if nprobe is not None and hasattr(index, "nprobe"):
        index.nprobe = nprobe

    scores, indices = index.search(query_emb, top_k)
    return indices[0], scores[0]


# ---------------------------
# Sparse Retrieval
# ---------------------------


def sparse_retrieval(
    query: str,
    bm25: BM25Okapi,
    *,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sparse retrieval using BM25.

    Args:
        query (str): The input query.
        bm25 (BM25Okapi): BM25 boject.
        top_k (int): The number of retrieved docs.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - indices: The indices of retrieved docs
            - scores[indices]: Th
    """
    scores = bm25.get_scores(query.split())
    indices = np.argsort(scores)[::-1][:top_k]
    return indices, scores[indices]


# ---------------------------
# Reciprocal Rank Fusion (Top-k based)
# ---------------------------


def reciprocal_rank_fusion(
    dense_idx: np.ndarray,
    sparse_idx: np.ndarray,
    *,
    rrf_k: int = 60,
) -> dict[int, float]:
    """
    Fuse dense and sparse retrieval rankings using Reciprocal Rank Fusion.

    Args:
        dense_idx (np.ndarray): Ranked document indices from dense retrieval.
        sparse_idx (np.ndarray): Ranked document indices from sparse retrieval.
        rrf_k (int): RRF scaling constant added to each rank.

    Returns:
        A mapping from document index to fused RRF score.
    """
    fused_scores: dict[int, float] = {}

    # Dense contribution
    for rank, doc_id in enumerate(dense_idx, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    # Sparse contribution
    for rank, doc_id in enumerate(sparse_idx, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (rrf_k + rank))

    return fused_scores


# ---------------------------
# Hybrid Retrieval
# ---------------------------


def hybrid_retrieval(
    query: str,
    query_emb: np.ndarray,
    faiss_index: faiss.Index,
    bm25: BM25Okapi,
    *,
    top_k: int = 10,
    dense_k: int = 50,
    sparse_k: int = 50,
    nprobe: int | None = None,
    rrf_k: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hybrid retrieval by fusing dense and sparse rankings.

    Args:
        query (str): The user query string.
        query_emb (np.ndarray): The dense embedding vector for the query.
        faiss_index (faiss.Index): FAISS index for dense retrieval.
        bm25 (BM25Okapi): BM25Okapi index for sparse retrieval.
        top_k (int): Number of final documents to return.
        dense_k (int): Number of dense candidates to retrieve before fusion.
        sparse_k (int): Number of sparse candidates to retrieve before fusion.
        nprobe (int | None): Optional number of FAISS partitions to search.
        rrf_k (int): Reciprocal Rank Fusion constant.

    Returns:
        A tuple of (final_idx, final_scores) for the fused top-k results.
    """

    # Retrieve candidates from both retrieval methods
    dense_idx, _ = dense_retrieval(
        query_emb,
        faiss_index,
        top_k=dense_k,
        nprobe=nprobe,
    )

    sparse_idx, _ = sparse_retrieval(
        query,
        bm25,
        top_k=sparse_k,
    )

    # Use RRF for ranking
    fused_dict = reciprocal_rank_fusion(
        dense_idx,
        sparse_idx,
        rrf_k=rrf_k,
    )

    # Select final top_k
    sorted_items = sorted(
        fused_dict.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    final_idx = np.array([doc_id for doc_id, _ in sorted_items])
    final_scores = np.array([score for _, score in sorted_items])

    return final_idx, final_scores
