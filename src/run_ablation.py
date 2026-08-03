from __future__ import annotations

import json
import os
import numpy as np
from collections.abc import Iterable, Mapping, Sequence
from math import log2
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from .builder import Builder
from .chunk_filter import get_company_candidate_indices
from .config import BuilderConfig, EvalConfig, QueryConfig
from .llm import compute_embeddings
from .query import extract_cadidates_info
from .rag_db import RAGDatabase


SUPPORT_WEIGHTS: dict[str, float] = {
    "full": 2.0,
    "partial": 1.0,
}
RETRIEVAL_METHODS = ("sparse", "dense", "hybrid")
TOP_K = (1, 5, 10)


def load_json(path: Path) -> Any:
    """Load and deserialize a UTF-8 JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        Json value (Any): The deserialized JSON value.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: Path) -> None:
    """
    Serialize data as indented UTF-8 JSON, creating parent directories.

    Args:
        data (Any): JSON-serializable value to save.
        path (Path): Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_gold_chunk_id(doc_id: str, paragraph_id: int) -> str:
    """
    Convert a document and paragraph ID into the corpus chunk ID.

    Args:
        doc_id (str): Document identifier.
        paragraph_id (int): paragraph number within the document.

    Returns:
        gold_chunk_id (str): Formatted chunk ID in the form "{doc_id}_{paragraph_id:02d}".
    """
    return f"{doc_id}_{paragraph_id:02d}"


def make_query_embeddings(
    *,
    client: OpenAI,
    model: str,
    queries: Sequence[str],
    output_path: Path,
    batch_size: int,
) -> np.ndarray:
    """
    Load cached query embeddings or compute and persist them.

    Args:
        client (OpenAI): OpenAI client used when embeddings are not cached.
        model (str): Embedding model name.
        queries (Sequence[str]): Query texts in evaluation-dataset order.
        output_path (Path): ``.npy`` cache path.
        batch_size (int): Number of queries sent per embedding batch.

    Returns:
        embeddings (np.ndarray): A NumPy array containing one embedding per query.

    Raises:
        ValueError: If the cached or computed embedding count differs from the number of queries.
    """
    if output_path.is_file() and output_path.stat().st_size > 0:
        embeddings = np.load(output_path)
    else:
        embeddings = compute_embeddings(
            client=client,
            model=model,
            texts=queries,
            output_path=output_path,
            save_emb=True,
            batch_size=batch_size,
        )

    if len(embeddings) != len(queries):
        raise ValueError(
            "Query embedding count does not match query count: "
            f"{len(embeddings)} != {len(queries)}"
        )

    return embeddings


def extract_filter_cache(
    *,
    client: OpenAI,
    model: str,
    eval_dataset: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Load or generate query-filter metadata keyed by query ID.

    An older list-shaped cache is converted in memory to the current mapping
    format for backward compatibility.

    Args:
        client (OpenAI): OpenAI client used to extract uncached filter metadata.
        model (str): Model used for company and year extraction.
        eval_dataset (Sequence[Mapping[str, Any]]): Evaluation examples containing ``id`` and ``query``.
        output_path (Path): JSON cache path.

    Returns:
        filtered_cache (dict[str, dict[str, Any]]): Filter metadata keyed by evaluation query ID.

    Raises:
        ValueError: If a legacy cache has a different length from the dataset,
            or if a mapping cache is missing query IDs required by the dataset.
        TypeError: If the cache is neither a list nor a mapping.
    """
    if output_path.is_file() and output_path.stat().st_size > 0:
        cached = load_json(output_path)

        # Backward compatibility with the old position-based list cache.
        if isinstance(cached, list):
            if len(cached) != len(eval_dataset):
                raise ValueError("Filter cache length does not match dataset length.")
            return {
                item["id"]: {
                    "query": item["query"],
                    **filter_info,
                }
                for item, filter_info in zip(eval_dataset, cached, strict=True)
            }

        if not isinstance(cached, Mapping):
            raise TypeError("Filter cache must be a JSON object or legacy list.")

        missing_ids = [item["id"] for item in eval_dataset if item["id"] not in cached]
        if missing_ids:
            preview = ", ".join(map(str, missing_ids[:5]))
            raise ValueError(
                "Filter cache is missing evaluation query IDs: "
                f"{preview}{' ...' if len(missing_ids) > 5 else ''}"
            )

        return dict(cached)

    results: dict[str, dict[str, Any]] = {}
    for item in tqdm(eval_dataset, desc="Extracting filters"):
        companies, start_year, end_year = extract_cadidates_info(
            client=client,
            query=item["query"],
            model=model,
        )
        results[item["id"]] = {
            "query": item["query"],
            "companies": companies,
            "start_year": start_year,
            "end_year": end_year,
        }

    save_json(results, output_path)
    return results


def retrieve_all(
    *,
    db: RAGDatabase,
    eval_dataset: Sequence[Mapping[str, Any]],
    query_embeddings: np.ndarray,
    filter_cache: Mapping[str, Mapping[str, Any]],
    company_info: Mapping[str, Any],
    query_config: QueryConfig,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """
    Run retrieval separately for every cutoff and ablation condition.

    Args:
        db (RAGDatabase): Loaded retrieval database.
        eval_dataset (Sequence[Mapping[str, Any]]): Evaluation examples containing IDs and query text.
        query_embeddings (np.ndarray): Query embeddings aligned with ``eval_dataset``.
        filter_cache (Mapping[str, Mapping[str, Any]]): Extracted company and year constraints keyed by query ID.
        company_info (Mapping[str, Any]): Metadata used to map extracted constraints to chunks.
        query_config (QueryConfig): Retrieval and rank-fusion settings.

    Returns:
        Nested results organized by condition, retrieval method, cutoff, and query ID.
    """
    results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        condition: {
            method: {f"top_{k}": {} for k in TOP_K} for method in RETRIEVAL_METHODS
        }
        for condition in ("unfiltered", "filtered")
    }

    for item, query_embedding in tqdm(
        zip(eval_dataset, query_embeddings, strict=True),
        total=len(eval_dataset),
        desc="Retrieving",
    ):
        query_id = item["id"]
        query = item["query"]
        filter_info = filter_cache[query_id]

        candidate_indices = get_company_candidate_indices(
            extracted_company=filter_info["companies"],
            start_year=filter_info["start_year"],
            end_year=filter_info["end_year"],
            company_info=company_info,
            chunk_ids=db.chunk_ids,
        )

        for method in RETRIEVAL_METHODS:
            for k in TOP_K:
                for condition, indices in (
                    ("unfiltered", None),
                    ("filtered", candidate_indices),
                ):
                    chunks = db.retrieve(
                        query=query,
                        query_emb=query_embedding,
                        top_k=k,
                        dense_k=query_config.dense_k,
                        sparse_k=query_config.sparse_k,
                        rrf_k=query_config.rrf_k,
                        candidate_indices=indices,
                        retrieval_method=method,
                    )

                    results[condition][method][f"top_{k}"][query_id] = {
                        "query": query,
                        "ranked_results": [
                            {
                                "chunk_id": chunk["chunk_id"],
                                "retrieval_score": float(chunk["score"]),
                            }
                            for chunk in chunks
                        ],
                    }

    return results


def build_gold_relevance(
    evidence_items: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Map each unique gold chunk to its highest annotated relevance grade.

    Args:
        evidence_items (Sequence[Mapping[str, Any]]): Gold evidence annotations with document,
            paragraph, and support labels.

    Returns:
        dict[str, float]: A mapping from corpus chunk ID to graded relevance.

    Raises:
        ValueError: If an evidence item has an unknown support label.
    """
    gold_relevance: dict[str, float] = {}

    for evidence in evidence_items:
        chunk_id = get_gold_chunk_id(
            evidence["doc_id"],
            evidence["paragraph_id"],
        )
        support = evidence["supports"]
        if support not in SUPPORT_WEIGHTS:
            raise ValueError(f"Unknown support label: {support!r}")

        gold_relevance[chunk_id] = max(
            gold_relevance.get(chunk_id, 0.0),
            SUPPORT_WEIGHTS[support],
        )

    return gold_relevance


def recall_at_k(
    gold_chunk_ids: Iterable[str],
    retrieved_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """
    Compute evidence-level Recall@K.

    Recall is the fraction of unique gold chunks present among the first ``k``
    retrieved chunks. Empty gold sets receive a score of 0.0.

    Args:
        gold_chunk_ids (Iterable[str]): Unique gold chunk IDs.
        retrieved_chunk_ids (Sequence[str]): Retrieved chunk IDs in rank order.
        k (int): Evaluation cutoff.

    Returns:
        Recall@K (float): Recall@K in the inclusive range [0.0, 1.0].
    """
    gold = set(gold_chunk_ids)
    if not gold:
        return 0.0

    retrieved = set(retrieved_chunk_ids[:k])
    return len(gold & retrieved) / len(gold)


def dcg(relevance_scores: Sequence[float]) -> float:
    """
    Compute discounted cumulative gain with exponential relevance gain.

    Args:
        relevance_scores (Sequence[float]): Relevance scores in rank order.

    Returns:
        DCG (float): Discounted cumulative gain.
    """
    return sum(
        (2.0**relevance - 1.0) / log2(rank + 2)
        for rank, relevance in enumerate(relevance_scores)
    )


def ideal_dcg_at_k(
    gold_relevance: Mapping[str, float],
    k: int,
) -> float:
    """
    Compute ideal DCG@K using chunk ID as a deterministic tie-breaker.

    Args:
        gold_relevance (Mapping[str, float]): Gold relevance grade keyed by chunk ID.
        k (int): Evaluation cutoff.

    Returns:
        Ideal DCG@K (float): Ideal DCG@K in the inclusive range [0.0, 1.0].
    """
    ideal_ranking = sorted(
        gold_relevance.items(),
        key=lambda pair: (-pair[1], pair[0]),
    )
    ideal_scores = [score for _, score in ideal_ranking[:k]]
    return dcg(ideal_scores)


def ndcg_at_k(
    gold_relevance: Mapping[str, float],
    retrieved_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """
    Compute graded NDCG@K from gold support labels.

    Args:
        gold_relevance (Mapping[str, float]): Gold relevance grade keyed by chunk ID.
        retrieved_chunk_ids (Sequence[str]): Retrieved chunk IDs in rank order.
        k (int): Evaluation cutoff.

    Returns:
        NDCG@K (float): NDCG@K in the inclusive range [0.0, 1.0].
    """
    actual_scores = [
        gold_relevance.get(chunk_id, 0.0) for chunk_id in retrieved_chunk_ids[:k]
    ]
    denominator = ideal_dcg_at_k(gold_relevance, k)
    if denominator == 0.0:
        return 0.0

    # Clamp tiny floating-point overshoots while preserving real errors.
    score = dcg(actual_scores) / denominator
    return min(max(score, 0.0), 1.0)


def mean(values: Sequence[float]) -> float:
    """
    Return the arithmetic mean, or 0.0 for an empty sequence.

    Args:
        values (Sequence[float]): A sequence of numeric values.

    Returns:
        Mean (float): The arithmetic mean of the values, or 0.0 if the sequence is empty.
    """
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval(
    *,
    eval_dataset: Sequence[Mapping[str, Any]],
    retrieved_chunks: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """
    Aggregate Recall@K and NDCG@K across the evaluation dataset.

    Args:
        eval_dataset (Sequence[Mapping[str, Any]]): Evaluation examples with IDs and gold evidence.
        retrieved_chunks (Mapping[str, Any]): Output produced by `retrieve_all`.

    Returns:
        Mean metrics organized by retrieval method, cutoff, and filter
        condition. Scores are rounded to four decimal places.
    """
    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for method in RETRIEVAL_METHODS:
        results[method] = {}

        for k in TOP_K:
            metrics = {
                condition: {"recall": [], "ndcg": []}
                for condition in ("unfiltered", "filtered")
            }

            for item in eval_dataset:
                query_id = item["id"]
                gold_relevance = build_gold_relevance(item["evidence"])

                for condition in ("unfiltered", "filtered"):
                    ranked_results = retrieved_chunks[condition][method][f"top_{k}"][
                        query_id
                    ]["ranked_results"]
                    retrieved_ids = [result["chunk_id"] for result in ranked_results]

                    metrics[condition]["recall"].append(
                        recall_at_k(gold_relevance.keys(), retrieved_ids, k)
                    )
                    metrics[condition]["ndcg"].append(
                        ndcg_at_k(gold_relevance, retrieved_ids, k)
                    )

            results[method][f"top_{k}"] = {
                condition: {
                    metric_name: round(mean(metric_values), 4)
                    for metric_name, metric_values in condition_metrics.items()
                }
                for condition, condition_metrics in metrics.items()
            }

    return results


def main() -> None:
    """Run database preparation, retrieval ablations, and metric evaluation."""

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    eval_config = EvalConfig()
    builder_config = BuilderConfig()
    query_config = QueryConfig()
    output_dir = Path(eval_config.output_dir)

    eval_dataset = load_json(Path(eval_config.dataset_path))

    db_name = "test"
    builder = Builder(cfg=builder_config, client=client)
    builder.build_database(db_name=db_name)
    db = RAGDatabase.load(db_name=db_name)

    queries = [item["query"] for item in eval_dataset]
    query_embeddings = make_query_embeddings(
        client=client,
        model=builder_config.embedding.model,
        queries=queries,
        output_path=output_dir / "query_embeddings.npy",
        batch_size=builder_config.embedding.batch_size,
    )

    filter_cache = extract_filter_cache(
        client=client,
        model=query_config.model,
        eval_dataset=eval_dataset,
        output_path=output_dir / "extracted_candidates_info.json",
    )

    company_info = load_json(Path(query_config.company_info_path))
    retrieval_path = output_dir / "retrieved_chunks_by_k.json"

    if retrieval_path.is_file() and retrieval_path.stat().st_size > 0:
        retrieved_chunks = load_json(retrieval_path)
    else:
        retrieved_chunks = retrieve_all(
            db=db,
            eval_dataset=eval_dataset,
            query_embeddings=query_embeddings,
            filter_cache=filter_cache,
            company_info=company_info,
            query_config=query_config,
        )
        save_json(retrieved_chunks, retrieval_path)

    eval_results = evaluate_retrieval(
        eval_dataset=eval_dataset,
        retrieved_chunks=retrieved_chunks,
    )
    save_json(eval_results, output_dir / "eval_results.json")


if __name__ == "__main__":
    main()
