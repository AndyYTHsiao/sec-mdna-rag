import json
from openai import OpenAI
from typing import Any
from .llm import generate_response
from .prompts import PROMPT_HELPER
from .config import QueryConfig
from .rag_db import RAGDatabase
from .chunk_filter import get_company_candidate_indices


def run_query(
    query_cfg: QueryConfig,
    client: OpenAI,
    query: str,
    db_name: str,
    *,
    registry_dir: str = "./artifacts/registry",
    company_info: dict[str, dict[str, str]] | None = None,
    fuzzy_threshold: float = 0.9,
) -> dict[str, Any]:
    """
    Run a retrieval-augmented generation query against a named database.

    Args:
        query_cfg (QueryConfig): Query configuration parameters for retrieval and LLM generation.
        client (OpenAI): OpenAI client used to generate responses.
        query (str): The user question to answer.
        db_name (str): Name of the database to load from the registry.
        registry_dir (str): Optional path to the database registry directory.
        company_info (dict[str, dict[str, str]] | None): Company metadata.
        fuzzy_threshold (float): Minimum similarity threshold for fuzzy company name matching.

    Returns:
        A dictionary containing:
            - "answer": the generated response text.
            - "retrieved_docs": the list of retrieved document chunks.
    """
    # Load DB
    db = RAGDatabase.load(db_name, registry_dir)

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

        filter_prompts = PROMPT_HELPER["filter_chunk"]
        filter_messages = _build_messages(
            filter_prompts["system"],
            filter_prompts["user"],
            query=query,
        )
        raw_filter_response = generate_response(
            client,
            filter_messages,
            query_cfg.model,
        )
        try:
            extracted_info = json.loads(raw_filter_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The filter extraction response was not valid JSON. "
                f"Response: {raw_filter_response!r}"
            ) from exc

        required_fields = {"companies", "start_year", "end_year"}
        missing_fields = required_fields - extracted_info.keys()

        if missing_fields:
            raise ValueError(
                "The filter extraction response is missing required fields: "
                f"{sorted(missing_fields)}"
            )

        candidate_indices = get_company_candidate_indices(
            extracted_info["companies"],
            extracted_info["start_year"],
            extracted_info["end_year"],
            company_info,
            chunk_ids=db.chunk_ids,
            fuzzy_threshold=fuzzy_threshold,
        )

    # Retrieve
    results = db.retrieve(query, client, query_cfg, candidate_indices=candidate_indices)

    # Build context
    context = _build_context(results) if results else "No relevant documents retrieved."

    # Build prompt
    query_prompts = PROMPT_HELPER["query_db"]
    messages = _build_messages(
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
        str: A formatted string containing document headers and text blocks,
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


def _build_messages(
    system_prompt: str, user_prompt: str, **kwargs
) -> list[dict[str, str]]:
    """
    Create chat messages for the LLM prompt from a query and context.

    Args:
        system_prompt (str): The system prompt.
        user_prompt (str): The user prompt.
        **kwargs: Additional arguments for prompts.

    Returns:
        Chat message (list[dict[str, str]]): A list of OpenAI chat message dictionaries with
        a system prompt and a user prompt.
    """
    try:
        user_content = user_prompt.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing prompt argument: {e.args[0]}") from e

    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_content.strip()},
    ]
