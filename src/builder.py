import json
import numpy as np
from dataclasses import asdict
from typing import Callable
from pathlib import Path
from openai import OpenAI
from .corpus import document_chunking_stream
from .llm import compute_embeddings
from .indexing import (
    build_faiss_index,
    build_bm25,
)
from .config import BuilderConfig
from .utils import load_corpus


class Builder:
    """
    Build RAG artifacts and register a searchable database.

    This class coordinates corpus chunking, embedding generation, FAISS index
    construction, BM25 index building, and registry persistence.
    """

    def __init__(
        self,
        cfg: BuilderConfig,
        client: OpenAI,
    ) -> None:
        """
        Initialize the builder with builder configuration and OpenAI client.

        Args:
            cfg (BuildConfig): Builder configuration values.
            client (OpenAI): OpenAI client used for embedding generation.
        """
        self.cfg = cfg
        self.client = client
        self.paths_cfg = self.cfg.paths

        artifacts = self.cfg.build_artifacts()
        self.hashes = {k: v.compute_hash() for k, v in artifacts.items()}

    def _artifact_path(self, key: str, base_dir: str, prefix: str, suffix: str) -> Path:
        """
        Return the versioned artifact path for the given key.

        Args:
            key (str): The artifact key (e.g., "corpus", "embedding").
            base_dir (str): The base directory for the artifact type.
            prefix (str): A prefix to identify the artifact type in the filename.
            suffix (str): The file extension for the artifact.

        Returns:
            Path: The full Path to the artifact file, incorporating the config hash.
        """
        return Path(base_dir) / f"{prefix}_{self.hashes[key]}{suffix}"

    def _get_or_build(self, path: Path, build_fn: Callable, *args, **kwargs) -> Path:
        """
        Return an existing artifact path or build it if missing or empty.

        Args:
            path (Path): The artifact path.
            build_fn (Callable): The function to build the artifact.
            *args: Additional arguments for the function to build the artifact.
            **kwargs: Additional keyword arguments for the function to build the artifact.

        Returns:
            artifact path (Path): The artifact path.
        """
        if path.is_file() and path.stat().st_size > 0:
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        build_fn(path, *args, **kwargs)

        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Failed to build artifact: {path}")

        return path

    def _build_corpus(self, output_path: Path) -> None:
        """
        Build the corpus artifact by chunking source filings into JSONL.

        Args:
            output_path (Path): The path where the corpus will be saved.
        """
        document_chunking_stream(
            filings_dir=Path(self.paths_cfg.filings_dir),
            output_path=output_path,
            max_tokens=self.cfg.corpus.max_tokens,
            max_paragraphs=self.cfg.corpus.max_paragraphs,
        )

    def _prepare_corpus(self) -> Path:
        """
        Prepare the corpus file path and build corpus if it does not exist.

        Returns:
            corpus_path (Path): The corpus path.
        """
        corpus_path = self._artifact_path(
            "corpus", self.paths_cfg.corpus_dir, "corpus", ".jsonl"
        )
        return self._get_or_build(corpus_path, self._build_corpus)

    def _build_embeddings(self, output_path: Path, texts: list[str]) -> None:
        """
        Compute and save dense embeddings for the provided text corpus.

        Args:
            output_path (Path): The path where the embeddings will be saved.
            texts (list[str]): The list of corpus texts.
        """
        compute_embeddings(
            self.client,
            self.cfg.embedding.model,
            texts,
            output_path,
            save_emb=True,
            batch_size=self.cfg.embedding.batch_size,
        )

    def _prepare_embeddings(self, corpus_path: Path) -> tuple[Path, list[str]]:
        """
        Prepare the embedding artifact path and return path plus corpus texts.

        Args:
            corpus (Path): The path where the corpus is saved.

        Returns:
            tuple[Path, list[str]]: The path of embeddings and a list of corpus texts.
        """
        emb_path = self._artifact_path(
            "embedding", self.paths_cfg.embeddings_dir, "embedding", ".npy"
        )

        _, texts = load_corpus(corpus_path)

        path = self._get_or_build(emb_path, self._build_embeddings, texts)
        return path, texts

    def _build_faiss(self, output_path: Path, emb_path: Path) -> None:
        """
        Build and save a FAISS index from precomputed embeddings.

        Args:
            output_path (Path): The path where the FAISS index will be saved.
            emb_path (Path): The path where the embeddings are saved.
        """
        embeddings = np.load(emb_path)
        build_faiss_index(
            embeddings,
            output_path=output_path,
            save_index=True,
        )

    def _prepare_faiss(self, emb_path: Path) -> Path:
        """
        Prepare the FAISS index artifact and build it if missing.

        Args:
            emb_path (Path): The path to the embedding.

        Returns:
            faiss_index_path (Path): The path to the FAISS index.
        """
        faiss_index_path = self._artifact_path(
            "faiss", self.paths_cfg.indexes_dir, "faiss", ".index"
        )

        return self._get_or_build(faiss_index_path, self._build_faiss, emb_path)

    def _build_bm25(self, output_path: Path, texts: list[str]) -> None:
        """
        Build and save a BM25 index from the corpus texts.

        Args:
            output_path (Path): The path where the BM25 index will be saved.
            texts (list[str]): The list of text chunks to index with BM25.
        """
        build_bm25(
            texts,
            k1=self.cfg.bm25.k1,
            b=self.cfg.bm25.b,
            output_path=output_path,
            save_bm25=True,
        )

    def _prepare_bm25(self, texts: list[str]) -> Path:
        """
        Prepare the BM25 object path and build it if missing.

        Args:
            texts (list[str]): The list of text chunks corresponding to chunk_ids.

        Returns:
            bm25_path (Path): The path to BM25 object.
        """
        bm25_path = self._artifact_path(
            "bm25", self.paths_cfg.indexes_dir, "bm25", ".pkl"
        )

        return self._get_or_build(bm25_path, self._build_bm25, texts)

    def build_database(
        self, db_name: str, registry_dir: str = "./artifacts/registry"
    ) -> None:
        """Build the database artifacts and write a registry entry.

        Args:
            db_name (str): The name to assign to the built database.
            registry_dir (str): Directory to store the registry JSON file.
        """
        corpus_path = self._prepare_corpus()
        emb_path, texts = self._prepare_embeddings(corpus_path)

        self._prepare_faiss(emb_path)
        self._prepare_bm25(texts)

        self._save_registry(db_name, Path(registry_dir))

    def _save_registry(self, db_name: str, registry_dir: Path) -> None:
        """
        Save the database registry file containing config and artifact hashes.

        Args:
            db_name (str): The name to assign to the built database.
            registry_dir (Path): Directory to store the registry JSON file.
        """
        registry_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "name": db_name,
            "config": asdict(self.cfg),
            "hashes": self.hashes,
        }

        with open(registry_dir / f"{db_name}.json", "w") as f:
            json.dump(entry, f, indent=4)
