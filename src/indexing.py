import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path


# ==============================
# BM25 (Sparse)
# ==============================


def tokenize(corpus: list[str]) -> list[list[str]]:
    """Tokenize the input corpus.

    Args:
        corpus (list[str]): The input corpus.

    Returns:
        list[list[str]]: Tokenized corpus (a list of token lists).
    """
    return [doc.split() for doc in corpus]


def build_bm25(
    corpus: list[str],
    k1: float = 1.5,
    b: float = 0.5,
    output_path: str | None = None,
    save_bm25: bool = False,
) -> BM25Okapi:
    """Build a BM25 index from a text corpus.

    The corpus is tokenized using the local `tokenize` function and then used
    to initialize a `BM25Okapi` model.

    Args:
        corpus (list[str]): A list of documents, where each document is a string.
        k1 (float): Term frequency scaling parameter (higher values increase
            the influence of term frequency).
        b (float): Document length normalization parameter (0 disables
            normalization, 1 fully normalizes by document length).
        output_path (str | None): If provided and `save_bm25` is True, the
            file path to write the pickled BM25 model to.
        save_bm25 (bool): If True, serialize the BM25 model to `output_path`.

    Returns:
        BM25Okapi: A BM25 model built from the tokenized corpus.
    """
    if not corpus:
        raise ValueError("`corpus` must contain at least one document")

    tokenized = tokenize(corpus)
    bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    if save_bm25:
        if output_path is None:
            raise ValueError("`output_path` must be provided when save_bm25=True")

        with open(output_path, "wb") as f:
            pickle.dump(bm25, f)

    return bm25


def load_bm25(path: Path) -> BM25Okapi:
    """
    Load a serialized BM25 model from disk.

    Args:
        path (Path): File path to the pickled BM25 model.

    Returns:
        BM25Okapi: The deserialized BM25 model.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ==============================
# FAISS (Dense)
# ==============================


def build_faiss_index(
    embeddings: np.ndarray,
    *,
    index_type: str = "flat",
    metric: str = "ip",
    nlist: int = 1024,
    output_path: str | Path | None = None,
    save_index: bool = False,
) -> faiss.Index:
    """Build and populate a FAISS index from embeddings.

    This helper is a convenience wrapper around FAISS index creation and
    optional persistence.

    Args:
        embeddings (np.ndarray): Array of shape (n, dim).
        index_type (str): Index type: "flat" or "ivf".
        metric (str): Distance metric: "ip" (inner product) or "l2" (L2 distance).
        nlist (int): Number of IVF clusters (used only for IVF).
        output_path (str | Path | None): If provided and `save_index` is True,
            the path to write the FAISS index to.
        save_index (bool): If True, the created index is written to `output_path`.

    Returns:
        faiss.Index: A trained and populated FAISS index.

    Notes:
        - embeddings must be shape (n, dim) and C-contiguous.
        - for metric="ip", embeddings should be L2-normalized if cosine similarity is intended.
    """
    assert embeddings.ndim == 2
    assert embeddings.flags["C_CONTIGUOUS"]

    dim = embeddings.shape[1]

    if metric == "ip":
        metric_type = faiss.METRIC_INNER_PRODUCT
        flat_cls = faiss.IndexFlatIP
    elif metric == "l2":
        metric_type = faiss.METRIC_L2
        flat_cls = faiss.IndexFlatL2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if index_type == "flat":
        index = flat_cls(dim)

    elif index_type == "ivf":
        if embeddings.shape[0] < nlist:
            raise ValueError(
                f"nlist ({nlist}) must be <= number of vectors ({embeddings.shape[0]})"
            )

        quantizer = flat_cls(dim)
        index = faiss.IndexIVFFlat(
            quantizer,
            dim,
            nlist,
            metric_type,
        )
        index.train(embeddings)

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(embeddings)

    if save_index:
        if output_path is None:
            raise ValueError("`output_path` must be provided when save_index=True")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path))

    return index


def load_faiss_index(path: str) -> faiss.Index:
    """
    Load FAISS index.

    Args:
        path (str): The path to FAISS index.

    Returns:
        faiss.Index: The saved FAISS index.
    """
    return faiss.read_index(path)
