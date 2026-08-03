import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from .utils import tokenize

# ---------------------------
# Dense Retrieval
# ---------------------------


def dense_retrieval(
    query_emb: np.ndarray,
    index: faiss.Index,
    *,
    top_k: int = 10,
    candidate_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve chunks using a FAISS inner-product index.

    The FAISS index is built over the full corpus. When filtering is enabled
    and candidate indices are provided, the full index is searched before
    retaining only eligible candidates. This guarantees that candidates are
    not missed because of a retrieval cutoff.

    Results are returned in descending inner-product score order.

    Args:
        query_emb (np.ndarray): Query embedding with shape ``(dimension,)`` or ``(1, dimension)``.
        index (faiss.Index): FAISS index using ``faiss.IndexFlatIP``.
        top_k (int): Maximum number of results to return.
        filter_chunks (bool): Whether to restrict retrieval using ``candidate_indices``.
        candidate_indices (np.ndarray | None): Global corpus indices eligible for retrieval.
            - ``None`` means filtering is unavailable, so retrieval falls back
              to the full corpus.
            - An empty array means filtering succeeded but no chunks matched.
            - A non-empty array restricts retrieval to those corpus indices.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Retrieved corpus indices and corresponding inner-product scores,
            ordered by score in descending order.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("index must use inner-product similarity.")

    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    elif query_emb.ndim != 2 or query_emb.shape[0] != 1:
        raise ValueError("query_emb must have shape (dimension,) or (1, dimension).")

    if query_emb.shape[1] != index.d:
        raise ValueError(
            f"Query dimension {query_emb.shape[1]} does not match "
            f"index dimension {index.d}."
        )

    # FAISS expects contiguous float32 input.
    query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)

    if candidate_indices is not None:
        candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

        if candidate_indices.ndim != 1:
            raise ValueError("candidate_indices must be a one-dimensional array.")

        if np.any(candidate_indices < 0) or np.any(candidate_indices >= index.ntotal):
            raise ValueError(
                "candidate_indices must contain valid indices into the FAISS index."
            )

    # Filtering succeeded, but no chunks matched.
    if candidate_indices is not None and candidate_indices.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    if index.ntotal == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )

    # Searching the full index ensures that no eligible candidate is excluded
    # by an arbitrary pre-filter retrieval cutoff.
    search_k = (
        index.ntotal if candidate_indices is not None else min(top_k, index.ntotal)
    )

    scores, indices = index.search(query_emb, search_k)

    indices = indices[0]
    scores = scores[0]

    # FAISS can return -1 for missing neighbors in some index configurations.
    valid_mask = indices != -1
    indices = indices[valid_mask]
    scores = scores[valid_mask]

    if candidate_indices is not None:
        candidate_mask = np.isin(indices, candidate_indices)
        indices = indices[candidate_mask]
        scores = scores[candidate_mask]

    return indices[:top_k], scores[:top_k]


# ---------------------------
# Sparse Retrieval
# ---------------------------


def sparse_retrieval(
    query: str,
    bm25: BM25Okapi,
    *,
    top_k: int = 10,
    candidate_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve chunks using BM25.

    Args:
        query (str): Input query.
        bm25 (BM25Okapi): BM25 index built over the full corpus.
        top_k (int): Maximum number of results to return.
        candidate_indices (np.ndarray | None): Global corpus indices eligible for retrieval.

    Returns:
        tuple[np.ndarray, np.ndarray]: Ranked corpus indices and their BM25 scores.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    scores = bm25.get_scores(tokenize(query))
    sorted_indices = np.argsort(scores)[::-1]

    # If candidate matching failed
    if candidate_indices is None:
        top_indices = sorted_indices[:top_k]
        return top_indices, scores[top_indices]

    # Return an empty array when no chunks are matched
    if len(candidate_indices) == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=scores.dtype),
        )

    filtered_sorted_indices = sorted_indices[np.isin(sorted_indices, candidate_indices)]
    top_indices = filtered_sorted_indices[:top_k]

    return top_indices, scores[top_indices]


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
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative.")

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
    rrf_k: int = 60,
    candidate_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform hybrid retrieval by fusing dense and sparse rankings.

    Args:
        query (str): User query string.
        query_emb (np.ndarray): Dense query embedding.
        faiss_index (faiss.Index): FAISS index using ``faiss.IndexFlatIP``.
        bm25 (BM25Okapi): BM25 index for sparse retrieval.
        top_k (int): Number of final documents to return.
        dense_k (int): Number of dense results to include before fusion.
        sparse_k (int): Number of sparse results to include before fusion.
        rrf_k (int): Non-negative Reciprocal Rank Fusion constant.
        filter_chunks (bool): Whether to restrict retrieval using ``candidate_indices``.
        candidate_indices (np.ndarray | None): Global corpus indices eligible for retrieval.
            - ``None`` means filtering is unavailable, so retrieval falls back
              to the full corpus.
            - An empty array means filtering succeeded but no chunks matched.
            - A non-empty array restricts retrieval to those corpus indices.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Final corpus indices and corresponding RRF scores, ordered by
            descending RRF score.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    if dense_k <= 0:
        raise ValueError("dense_k must be greater than 0.")

    if sparse_k <= 0:
        raise ValueError("sparse_k must be greater than 0.")

    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative.")

    dense_idx, _ = dense_retrieval(
        query_emb,
        faiss_index,
        top_k=dense_k,
        candidate_indices=candidate_indices,
    )

    sparse_idx, _ = sparse_retrieval(
        query,
        bm25,
        top_k=sparse_k,
        candidate_indices=candidate_indices,
    )

    fused_scores = reciprocal_rank_fusion(
        dense_idx,
        sparse_idx,
        rrf_k=rrf_k,
    )

    sorted_items = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    final_idx = np.asarray(
        [doc_id for doc_id, _ in sorted_items],
        dtype=np.int64,
    )
    final_scores = np.asarray(
        [score for _, score in sorted_items],
        dtype=np.float32,
    )

    return final_idx, final_scores
