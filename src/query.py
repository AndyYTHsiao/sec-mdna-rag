from openai import OpenAI
from typing import Any
from .llm import generate_response, compute_embeddings, build_input_messages
from .prompts import PROMPT_HELPER
from .config import QueryConfig
from .rag_db import RAGDatabase
from .chunk_filter import extract_cadidates_info, get_company_candidate_indices


def run_query(
    query_cfg: QueryConfig,
    client: OpenAI,
    query: str,
    db_name: str,
    *,
    company_info: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Run a retrieval-augmented generation query against a named database.

    Args:
        query_cfg (QueryConfig): Query configuration for retrieval and LLM generation.
        client (OpenAI): OpenAI client used to generate responses.
        query (str): The user question to answer.
        db_name (str): Name of the database to load from the registry.
        company_info (dict[str, dict[str, str]] | None): Company metadata.


    Returns:
        A dictionary containing:
            - "answer": the generated response text.
            - "retrieved_docs": the list of retrieved document chunks.
    """
    # Load DB
    db = RAGDatabase.load(db_name)

    # Find candidate indices based on identified criteria
    candidate_indices = None
    if query_cfg.filter_chunks:
        if company_info is None:
            raise ValueError(
                "company_info is required when query_cfg.filter_chunks is enabled."
            )

        if db.texts is None:
            raise ValueError(
                "chunks is required when query_cfg.filter_chunks is enabled."
            )

        companies, start_year, end_year = extract_cadidates_info(
            client, query, query_cfg.model
        )
        candidate_indices = get_company_candidate_indices(
            companies,
            start_year,
            end_year,
            company_info,
            chunk_ids=db.chunk_ids,
            fuzzy_threshold=query_cfg.fuzzy_threshold,
        )

    # Retrieve
    query_emb = compute_embeddings(client, query_cfg.embedding_model, [query])
    results = db.retrieve(
        query,
        query_emb,
        top_k=query_cfg.top_k,
        dense_k=query_cfg.dense_k,
        sparse_k=query_cfg.sparse_k,
        rrf_k=query_cfg.rrf_k,
        candidate_indices=candidate_indices,
        retrieval_method=query_cfg.retrieval_method,
    )

    # Build context
    context = _build_context(results) if results else "No relevant documents retrieved."

    # Build prompt
    query_prompts = PROMPT_HELPER["query_db"]
    messages = build_input_messages(
        query_prompts["system"], query_prompts["user"], context=context, query=query
    )

    # Generate answer
    answer = generate_response(client, messages, query_cfg.model)

    return {
        "answer": answer,
        "retrieved_docs": results,
    }


def _build_context(results: list[dict], max_chars: int = 6000) -> str:
    """
    Build a prompt context string from retrieved document chunks.

    Args:
        results (list[dict]): A list of retrieved document metadata dictionaries.
            Each item must contain 'chunk_id', 'score', and 'text'.
        max_chars (int): Maximum number of characters to include in the
            returned context string.

    Returns:
        formatted context (str): A formatted string containing document headers and text blocks,
            joined by separators. The returned string is truncated safely
            once adding the next block would exceed max_chars.
    """
    context_blocks = []
    total_chars = 0

    for rank, doc in enumerate(results, start=1):
        block = (
            f"[Document {rank} | ID: {doc['chunk_id']} | Score: {doc['score']:.4f}]\n"
            f"{doc['text'].strip()}\n"
        )

        if total_chars + len(block) > max_chars:
            break

        context_blocks.append(block)
        total_chars += len(block)

    return "\n---\n".join(context_blocks)
