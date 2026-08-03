import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any
from faiss import read_index, Index
from rank_bm25 import BM25Okapi
from .retrieval import sparse_retrieval, dense_retrieval, hybrid_retrieval
from .utils import load_corpus


class RAGDatabase:
    """
    This class holds the loaded corpus metadata, dense/faiss index, and
    sparse/bm25 index needed to retrieve relevant chunks for a query.
    """

    def __init__(
        self,
        name: str,
        embedding_model: str,
        chunk_ids: list[str],
        texts: list[str],
        faiss_index: Index,
        bm25_index: BM25Okapi,
    ) -> None:
        """
        Initialize a loaded RAG database instance.

        Args:
            name (str): The database name.
            embedding_model (str): The embedding model used for queries.
            chunk_ids (list[str]): The list of chunk identifiers for the corpus.
            texts (list[str]): The list of text chunks corresponding to chunk_ids.
            faiss_index (Index): The dense FAISS index used for semantic retrieval.
            bm25_index (BM25Okapi): The sparse BM25 index used for lexical retrieval.
        """
        corpus_size = len(chunk_ids)

        if len(texts) != corpus_size:
            raise ValueError(
                "chunk_ids and texts must have the same length: "
                f"{corpus_size} != {len(texts)}."
            )

        if faiss_index.ntotal != corpus_size:
            raise ValueError(
                "FAISS index size must match the corpus size: "
                f"{faiss_index.ntotal} != {corpus_size}."
            )

        if len(bm25_index.doc_len) != corpus_size:
            raise ValueError(
                "BM25 index size must match the corpus size: "
                f"{len(bm25_index.doc_len)} != {corpus_size}."
            )

        self.name = name
        self.embedding_model = embedding_model
        self.chunk_ids = chunk_ids
        self.texts = texts
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index

    @classmethod
    def load(
        cls, db_name: str, registry_dir: str = "./artifacts/registry"
    ) -> "RAGDatabase":
        """
        Load a database from its registry entry and return a RAGDatabase.

        Args:
            db_name (str): The name of the database to load.
            registry_dir (str): Directory containing database registry JSON files.

        Returns:
           RAGDatabase: A fully initialized RAGDatabase with corpus data and indexes loaded.

        Raises:
            ValueError: If the requested database registry file does not exist.
        """
        registry_path = Path(registry_dir) / f"{db_name}.json"

        if not registry_path.exists():
            raise ValueError(f"Database '{db_name}' not found.")

        with open(registry_path) as f:
            db_registry = json.load(f)

        paths = db_registry["config"]["paths"]
        hashes = db_registry["hashes"]

        corpus_path = Path(paths["corpus_dir"]) / f"corpus_{hashes['corpus']}.jsonl"

        index_dir = Path(paths["indexes_dir"])
        faiss_path = index_dir / f"faiss_{hashes['faiss']}.index"
        bm25_path = index_dir / f"bm25_{hashes['bm25']}.pkl"

        embedding_model = db_registry["config"]["embedding"]["model"]

        chunk_ids, texts = load_corpus(corpus_path)
        faiss_index = read_index(str(faiss_path))

        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)

        return cls(
            name=db_name,
            embedding_model=embedding_model,
            chunk_ids=chunk_ids,
            texts=texts,
            faiss_index=faiss_index,
            bm25_index=bm25_index,
        )

    def retrieve(
        self,
        query: str,
        query_emb: np.ndarray,
        *,
        top_k: int,
        dense_k: int,
        sparse_k: int,
        rrf_k: int,
        candidate_indices: np.ndarray | None = None,
        retrieval_method: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """
        Retrieve ranked document chunks for a query using hybrid search.

        Args:
            query (str): The user query string.
            query_emb (np.ndarray): Query embedding with shape ``(dimension,)`` or ``(1, dimension)``.
            top_k (int): Number of final documents to return.
            dense_k (int): Number of dense results to include before fusion.
            sparse_k (int): Number of sparse results to include before fusion.
            rrf_k (int): Non-negative Reciprocal Rank Fusion constant.
            candidate_indices (np.ndarray | None): Global corpus indices eligible for retrieval.
                - ``None`` means filtering is unavailable, so retrieval falls back
                    to the full corpus.
                - An empty array means filtering succeeded but no chunks matched.
                - A non-empty array restricts retrieval to those corpus indices.
            retrieval_method (str): Retrieval method to use: "sparse", "dense", or "hybrid".

        Returns:
            list[dict[str, Any]]: A list of dictionary containing retrieved result.
                - chunk IDs (str)
                - text (str)
                - score (float)
        """
        if retrieval_method not in {"sparse", "dense", "hybrid"}:
            raise ValueError(
                "Unsupported methods. Choose 'sprase', 'dense', or 'hybrid'"
            )

        elif retrieval_method == "sparse":
            indices, scores = sparse_retrieval(
                query=query,
                bm25=self.bm25_index,
                top_k=top_k,
                candidate_indices=candidate_indices,
            )

        elif retrieval_method == "dense":
            indices, scores = dense_retrieval(
                query_emb=query_emb,
                index=self.faiss_index,
                top_k=top_k,
                candidate_indices=candidate_indices,
            )

        elif retrieval_method == "hybrid":
            indices, scores = hybrid_retrieval(
                query,
                query_emb,
                self.faiss_index,
                self.bm25_index,
                top_k=top_k,
                dense_k=dense_k,
                sparse_k=sparse_k,
                rrf_k=rrf_k,
                candidate_indices=candidate_indices,
            )

        return [
            {
                "chunk_id": self.chunk_ids[i],
                "text": self.texts[i],
                "score": float(score),
            }
            for i, score in zip(indices, scores, strict=True)
        ]
