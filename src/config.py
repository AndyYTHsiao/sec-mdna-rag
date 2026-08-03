from dataclasses import (
    dataclass,
    asdict,
    field,
    fields,
    MISSING,
    is_dataclass,
)
from typing import Any, Type, TypeVar
from .utils import hash_config

T = TypeVar("T")


def build_dataclass(cls: Type[T], data: dict[str, Any]) -> T:
    """
    Construct an instance of a dataclass from a dictionary.

    This function iterates over the fields defined in the dataclass
    ``cls`` and populates them using values from ``data`` when available.
    If a field is not present in ``data``, its default value or
    ``default_factory`` is used if defined. If neither is available,
    an error is raised.

    Extra keys in ``data`` that do not correspond to fields in ``cls``
    are ignored.

    Args:
        cls (Type[T]): The dataclass type to instantiate.
        data (dict[str, Any]): A dictionary containing values to populate the dataclass fields.
                            Keys should match field names defined in ``cls``.

    Returns:
        An instance of the dataclass (T): ``cls`` populated with values from
        ``data`` and default values where necessary.

    Raises:
        ValueError: If a required field (one without a default value or
            ``default_factory``) is missing from ``data``.

    Notes:
    - Fields found in ``data`` take precedence over defaults.
    - ``default_factory`` is called to generate a value when defined.
    - The function assumes that ``cls`` is a valid dataclass type.
    - Type validation is not performed beyond what the dataclass
      constructor enforces.
    """

    kwargs: dict[str, Any] = {}

    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = data[f.name]
            continue

        if f.default is not MISSING:
            kwargs[f.name] = f.default
            continue

        # Some Field objects may not have default_factory attribute set; use getattr.
        default_factory = getattr(f, "default_factory", MISSING)
        if default_factory is not MISSING:
            kwargs[f.name] = default_factory()
            continue

        raise ValueError(f"Missing required field '{f.name}' for {cls.__name__}")

    return cls(**kwargs)


@dataclass(frozen=True)
class Artifact:
    name: str
    config: Any = None
    dependencies: tuple["Artifact", ...] = ()

    def compute_hash(self) -> str:
        dep_hashes = tuple(dep.compute_hash() for dep in self.dependencies)

        if is_dataclass(self.config):
            cfg = asdict(self.config)
        else:
            cfg = self.config

        payload = {
            "name": self.name,
            "config": cfg,
            "dependencies": dep_hashes,
        }

        return hash_config(payload)


@dataclass(frozen=True)
class PathsConfig:
    filings_dir: str = "./data/filings"
    corpus_dir: str = "./artifacts/corpus"
    embeddings_dir: str = "./artifacts/embeddings"
    indexes_dir: str = "./artifacts/indexes"
    registry_dir: str = "./artifacts/registry"


@dataclass(frozen=True)
class CorpusConfig:
    max_tokens: int = 700
    max_paragraphs: int = 1


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    batch_size: int = 256


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class EvalConfig:
    output_dir: str = "./artifacts/eval"
    dataset_path: str = "./data/eval_dataset.json"


@dataclass
class QueryConfig:
    model: str = "gpt-5.6-luna"
    embedding_model: str = "text-embedding-3-small"
    retrieval_method: str = "hybrid"
    top_k: int = 5
    dense_k: int = 5
    sparse_k: int = 5
    rrf_k: int = 60
    filter_chunks: bool = True
    fuzzy_threshold: float = 0.9
    company_info_path: str = "./data/company_metadata.json"


@dataclass(frozen=True)
class BuilderConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    bm25: BM25Config = field(default_factory=BM25Config)

    def build_artifacts(self) -> dict[str, Artifact]:
        corpus_artifact = Artifact(
            name="corpus",
            config=self.corpus,
        )

        embedding_artifact = Artifact(
            name="embedding",
            config=self.embedding,
            dependencies=(corpus_artifact,),
        )

        faiss_artifact = Artifact(
            name="faiss",
            config=None,
            dependencies=(embedding_artifact, corpus_artifact),
        )

        bm25_artifact = Artifact(
            name="bm25",
            config=self.bm25,
            dependencies=(corpus_artifact,),
        )

        return {
            "corpus": corpus_artifact,
            "embedding": embedding_artifact,
            "faiss": faiss_artifact,
            "bm25": bm25_artifact,
        }
