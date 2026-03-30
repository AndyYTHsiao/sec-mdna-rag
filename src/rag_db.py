import json
import pickle
from openai import OpenAI
from pathlib import Path
from typing import Any
from faiss import read_index, Index
from rank_bm25 import BM25Okapi
from llm import compute_embeddings
from retrieval import hybrid_retrieval
from utils import load_corpus
from config import QueryConfig


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
            embedding_model(str): The embedding model used for queries.
            chunk_ids (list[str]):: The list of chunk identifiers for the corpus.
            texts (list[str]): The list of text chunks corresponding to chunk_ids.
            faiss_inde (Index): The dense FAISS index used for semantic retrieval.
            bm25_index (BM25Okapi): The sparse BM25 index used for lexical retrieval.
        """
        self.name = name
        self.embedding_model = embedding_model
        self.chunk_ids = chunk_ids
        self.texts = texts
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index

    @classmethod
    def load(cls, db_name: str, registry_dir: str = "./artifacts/registry"):
        """
        Load a database from its registry entry and return a RAGDatabase.

        Args:
            db_name: The name of the database to load.
            registry_dir: Directory containing database registry JSON files.

        Returns:
            A fully initialized RAGDatabase with corpus data and indexes loaded.

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

        corpus_path = Path(paths["corpus_path"]) / f"corpus_{hashes['corpus']}.jsonl"

        index_dir = Path(paths["indexes_path"])
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
        self, query: str, client: OpenAI, query_cfg: QueryConfig
    ) -> list[dict[str, Any]]:
        """
        Retrieve ranked document chunks for a query using hybrid search.

        Args:
            query (str): The user query string.
            client (OpenAI): OpenAI client used to compute dense embeddings.
            query_cfg (QueryConfig): Query configuration controlling retrieval behavior.

        Returns:
            A list of dictionaries containing chunk_id, text, and score
            for each retrieved result.
        """
        query_emb = compute_embeddings(
            client,
            self.embedding_model,
            [query],
        )[0]

        idx, scores = hybrid_retrieval(
            query,
            query_emb,
            self.faiss_index,
            self.bm25_index,
            top_k=query_cfg.top_k,
            dense_k=query_cfg.dense_k,
            sparse_k=query_cfg.sparse_k,
            rrf_k=query_cfg.rrf_k,
        )

        return [
            {
                "chunk_id": self.chunk_ids[i],
                "text": self.texts[i],
                "score": float(scores[k]),
            }
            for k, i in enumerate(idx)
        ]
