import json
import questionary
import numpy as np
from dataclasses import asdict
from typing import Callable
from pathlib import Path
from openai import OpenAI
from corpus import document_chunking_stream
from llm import compute_embeddings
from indexing import (
    build_faiss_index,
    build_bm25,
)
from config import PipelineConfig
from utils import load_corpus


class Builder:
    """
    Build RAG artifacts and register a searchable database.

    This class coordinates corpus chunking, embedding generation, FAISS index
    construction, BM25 index building, and registry persistence.
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        client: OpenAI,
    ) -> None:
        """
        Initialize the builder with pipeline configuration and OpenAI client.

        Args:
            cfg (PipelineConfig): Pipeline configuration values.
            client (OpenAI): OpenAI client used for embedding generation.
        """
        self.cfg = cfg
        self.client = client
        self.paths = self.cfg.paths

        self.artifacts = self.cfg.build_artifacts()
        self.hashes = {k: v.compute_hash() for k, v in self.artifacts.items()}

    def _artifact_path(
        self, key: str, base_path: str, prefix: str, suffix: str
    ) -> Path:
        """
        Return the versioned artifact path for the given key.

        Args:
            key (str): The artifact key (e.g., "corpus", "embedding").
            base_path (str): The base directory for the artifact type.
            prefix (str): A prefix to identify the artifact type in the filename.
            suffix (str): The file extension for the artifact.

        Returns:
            The full Path to the artifact file, incorporating the config hash.
        """
        return Path(base_path) / f"{prefix}_{self.hashes[key]}{suffix}"

    def _get_or_build(self, path: Path, build_fn: Callable, *args, **kwargs) -> Path:
        """
        Return an existing artifact path or build it if missing or empty.

        Args:
            path (Path): The artifact path.
            build_fn (Callable): The function to build the artifact.
            *args: Additional arguments for the function to build the artifact.
            **kwargs: Additional keyword arguments for the function to build the artifact.

        Returns:
            The artifact path.
        """
        if path.exists() and path.stat().st_size > 0:
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        build_fn(*args, **kwargs)
        return path

    def _build_corpus(self, output_path: Path) -> None:
        """
        Build the corpus artifact by chunking source filings into JSONL.

        Args:
            output_path (Path): The path where the corpus will be saved.
        """
        document_chunking_stream(
            filings_dir=Path(self.cfg.corpus.filings_dir),
            output_path=output_path,
            max_tokens=self.cfg.corpus.max_tokens,
            max_paragraphs=self.cfg.corpus.max_paragraphs,
        )

    def _prepare_corpus(self) -> Path:
        """
        Prepare the corpus file path and build corpus if it does not exist.

        Returns:
            The corpus path.
        """
        path = self._artifact_path("corpus", self.paths.corpus_path, "corpus", ".jsonl")
        return self._get_or_build(path, self._build_corpus, path)

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
        )

    def _prepare_embeddings(self, corpus_path: Path) -> tuple[Path, list[str]]:
        """
        Prepare the embedding artifact path and return path plus corpus texts.

        Args:
            corpus (Path): The path where the corpus is saved.

        Returns:
            A tuple containing the path of embeddings and a list of corpus texts.
        """
        path = self._artifact_path(
            "embedding", self.paths.embeddings_path, "embedding", ".npy"
        )

        _, texts = load_corpus(corpus_path)

        path = self._get_or_build(path, self._build_embeddings, path, texts)
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
            index_type=self.cfg.faiss.index_type,
            metric=self.cfg.faiss.metric,
            nlist=self.cfg.faiss.nlist,
            output_path=output_path,
            save_index=True,
        )

    def _prepare_faiss(self, emb_path: Path) -> Path:
        """Prepare the FAISS index artifact and build it if missing."""
        path = self._artifact_path("faiss", self.paths.indexes_path, "faiss", ".index")

        return self._get_or_build(path, self._build_faiss, path, emb_path)

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
        Prepare the BM25 artifact path and build it if missing.

        Args:
            texts (list[str]): The list of text chunks corresponding to chunk_ids.

        Returns:
            The path to BM25 artifact.
        """
        path = self._artifact_path("bm25", self.paths.indexes_path, "bm25", ".pkl")

        return self._get_or_build(path, self._build_bm25, path, texts)

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

        questionary.print(f"\n[✓] Database '{db_name}' ready.", style="bold fg:green")

    def _save_registry(self, db_name: str, registry_dir: Path):
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
