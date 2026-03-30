from openai import OpenAI
from typing import Any
from llm import generate_response
from config import QueryConfig
from rag_db import RAGDatabase


def run_query(
    query_cfg: QueryConfig,
    client: OpenAI,
    query: str,
    db_name: str,
    *,
    registry_dir: str = "./artifacts/registry",
) -> dict[str, Any]:
    """
    Run a retrieval-augmented generation query against a named database.

    Args:
        query_cfg (QueryConfig): Query configuration parameters for retrieval and LLM generation.
        client (OpenAI): OpenAI client used to generate responses.
        query (str): The user question to answer.
        db_name (str): Name of the database to load from the registry.
        registry_dir (str): Optional path to the database registry directory.

    Returns:
        A dictionary containing:
            - "answer": the generated response text.
            - "retrieved_docs": the list of retrieved document chunks.
    """
    # Load DB
    db = RAGDatabase.load(db_name, registry_dir)

    # Retrieve
    results = db.retrieve(query, client, query_cfg)

    # Build context
    context = build_context(results) if results else "No relevant documents retrieved."

    # Build prompt
    messages = build_messages(query, context)

    # Generate answer
    answer = generate_response(client, messages, query_cfg.model)

    return {
        "answer": answer,
        "retrieved_docs": results,
    }


def build_context(results: list[dict], max_chars: int = 6000) -> str:
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


def build_messages(query: str, context: str) -> list[dict[str, str]]:
    """
    Create chat messages for the LLM prompt from a query and context.

    Args:
        query: The user's question.
        context: The retrieval context to provide to the model.

    Returns:
        A list of OpenAI chat message dictionaries with a system prompt
        and a user prompt containing the context and query.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions "
                "using the provided context.\n\n"
                "Follow these rules strictly:\n"
                "1. Carefully read the context before answering.\n"
                "2. If the answer exists in the context:\n"
                "   - Use the information from the context.\n"
                "   - Cite the document number.\n"
                "3. If the answer does NOT exist in the context:\n"
                "   - Answer from general knowledge.\n"
                "   - Start your answer with:\n"
                "     'This answer is based on my general knowledge.'\n"
                "4. Never fabricate citations.\n"
                "5. Prefer using context whenever possible.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{query}"
            ),
        },
    ]
