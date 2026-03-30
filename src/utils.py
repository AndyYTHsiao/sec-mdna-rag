import json
import hashlib
from pathlib import Path
from typing import Any


def load_corpus(file_path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Load chunk IDs and texts from corpus.
    Each row must contain: { "chunk_id": str, "text": str }.

    Args:
        file_path (str): The file path of chunked filings.

    Returns:
        list[str]: Chunk IDs.
        list[list[str]]: Chunked texts.
    """
    chunk_ids = []
    texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            if "chunk_id" not in row or "text" not in row:
                raise ValueError(f"Invalid row at line {line_num}")

            chunk_ids.append(row["chunk_id"])
            texts.append(row["text"])

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Duplicate chunk_ids detected")

    return chunk_ids, texts


def hash_config(obj: Any, *, algo: str = "sha256", length: int = 12) -> str:
    """
    Compute a short, stable hash for configs / metadata.

    Args:
        obj (Any): Any JSON-serializable object
        algo (str): Hash algorithm (default: sha256)
        length (int): Length of returned hex digest

    Returns:
        Short hash string (e.g. 'a3f91c2e4d7b')
    """
    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    h = hashlib.new(algo)
    h.update(payload)

    return h.hexdigest()[:length]


def list_existing_databases(db_dir: str = "./artifacts/registry") -> list[str]:
    """
    List existing databases.

    Args:
        db_dir (str): Directory of the databases.

    Returns:
        list[str]: List of databases.
    """
    return [p.stem for p in Path(db_dir).glob("*.json")]
