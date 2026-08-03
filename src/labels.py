"""
Human-readable labels for configuration dataclasses.

Used by:
- CLI prompts
- Streamlit UI
"""

# -----------------------------
# Dataclass Titles
# -----------------------------

CLASS_LABELS = {
    "PipelineConfig": "Pipeline Settings",
    "PathsConfig": "Path Settings",
    "CorpusConfig": "Corpus Processing Settings",
    "EmbeddingConfig": "Embedding Settings",
    "FaissConfig": "FAISS Index Settings",
    "BM25Config": "BM25 Index Settings",
    "QueryConfig": "Query & Retrieval Settings",
}


# -----------------------------
# Generic Field Labels
# (fallback if no qualified match)
# -----------------------------

FIELD_LABELS = {
    # Paths
    "filings_dir": "Raw Filings Directory",
    "corpus_dir": "Corpus Output Directory",
    "embeddings_dir": "Embeddings Output Directory",
    "indexes_dir": "Indexes Output Directory",
    "registry_dir": "Registry Output Directory",
    # Corpus
    "max_paragraphs": "Maximum Paragraphs per Chunk",
    # Embeddings
    "model": "Model Name",
    # BM25
    "k1": "BM25 k1 Parameter",
    "b": "BM25 b Parameter",
    # Query
    "embedding_model": "Embedding Model",
    "retrieval_method": "Retrieval Method",
    "top_k": "Final Retrieved Chunks",
    "dense_k": "Dense Retriever Top K",
    "sparse_k": "Sparse Retriever Top K",
    "rrf_k": "Reciprocal Rank Fusion k",
    "filter_chunks": "Whether to filter chunks",
    "fuzzy_threshold": "Minimum Similarity Threshold for Fuzzy Company Name Matching (0-1)",
    "company_info_path": "Path to Company Information JSON File",
}


# -----------------------------
# Qualified Labels
# (override generic names)
# -----------------------------

QUALIFIED_FIELD_LABELS = {
    # Embedding
    "EmbeddingConfig.model": "Embedding Model",
    "EmbeddingConfig.batch_size": "Batch Size per Request",
    # Query
    "QueryConfig.model": "LLM Model",
    "QueryConfig.top_k": "Final Top K Chunks",
    "QueryConfig.dense_k": "Dense Retrieval Top K",
    "QueryConfig.sparse_k": "Sparse Retrieval Top K",
    "QueryConfig.rrf_k": "RRF Constant (k)",
    "QueryConfig.filter_chunks": "Filter Chunks",
    "QueryConfig.fuzzy_threshold": "Minimum Similarity Threshold for Fuzzy Company Name Matching (0-1)",
    "QueryConfig.company_info_path": "Path to Company Information JSON File",
    # Corpus
    "CorpusConfig.max_tokens": "Chunk Size (tokens)",
}


# -----------------------------
# Help Support
# -----------------------------

FIELD_HELP = {
    "CorpusConfig.max_tokens": "Maximum number of tokens per chunk.",
    "FaissConfig.nlist": "Number of Voronoi cells used in IVF indexing.",
    "QueryConfig.rrf_k": "Higher values reduce the influence of lower-ranked results.",
}


def get_field_help(cls_name: str, field_name: str) -> str | None:
    """
    Return the help message of the given field.

    Args:
        cls_name (str): Dataclass name.
        field_name (str): Field name.

    Returns:
        Help message of the field.
    """
    qualified_key = f"{cls_name}.{field_name}"
    return FIELD_HELP.get(qualified_key)


def get_class_label(cls_name: str) -> str:
    """
    Return the class label.

    Args:
        cls_name (str): Dataclass name.

    Returns:
        The label for the dataclass.
    """
    return CLASS_LABELS.get(cls_name, cls_name)


def get_field_label(cls_name: str, field_name: str) -> str:
    """
    Return the field label.

    Args:
        cls_name (str): Dataclass name.
        field_name (str): Field name.

    Returns:
        The label for the field.
    """
    qualified_key = f"{cls_name}.{field_name}"

    return QUALIFIED_FIELD_LABELS.get(
        qualified_key,
        FIELD_LABELS.get(field_name, field_name),
    )
