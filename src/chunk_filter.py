import re
import json
import numpy as np
from openai import OpenAI
from difflib import SequenceMatcher
from .prompts import PROMPT_HELPER
from .llm import generate_response, build_input_messages

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "companies",
    "ltd",
    "limited",
    "plc",
}


def _normalize_company_text(name: str) -> str:
    """
    Normalize a company name.

    Args:
        name (str): The company name.

    Returns:
        str: Normalized company name.
    """
    name = name.lower().strip()
    name = re.sub(r"'s\b", "", name)
    name = name.replace("&", " and ")
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _remove_legal_suffixes(name: str) -> str:
    """
    Remove legal suffixes from a company name.

    Args:
        name (str): The company name.

    Returns:
        str: The company name without legal suffixes.
    """
    tokens = _normalize_company_text(name).split()

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def _build_company_lookup(
    company_info: dict[str, dict[str, str]],
) -> dict[str, str]:
    """
    Build alias lookup.

    Example:
    {
        "apple inc": "320193",
        "apple": "320193",
        "aapl": "320193"
    }

    Args:
        company_info (dict[str, dict[str, str]]): Company metadata.

    Returns:
        company lookup table (dict[str, str): The company lookup table mapping aliases to CIKs.
    """
    lookup = {}

    for cik, info in company_info.items():
        ticker = info["ticker"]
        name = info["name"]

        aliases = {
            _normalize_company_text(name),
            _remove_legal_suffixes(name),
            _normalize_company_text(ticker),
        }

        for alias in aliases:
            if alias:
                lookup[alias] = cik

    return lookup


def _similarity(a: str, b: str) -> float:
    """
    Calculate the similarity between two sequences.

    Args:
        a (str): Candidate sequence a.
        b (str): Candidate sequence b.

    Returns:
        Ratio (float): Similarity ratio between the two sequences.
    """
    return SequenceMatcher(None, a, b).ratio()


def _match_company_to_cik(
    extracted_company: list[str],
    company_lookup: dict[str, str],
    fuzzy_threshold: float = 0.9,
) -> set[str]:
    """
    Map the extracted company names to corresponding Central Index Keys (CIKs).

    Steps:
        1. Exact match: Check if the extracted name has an exact match in the lookup table.
        2. Fuzzy match: If an exact match does not exist, check if one of the aliases has
                        a similarity ratio higher than the threshold.

    Args:
        extracted_company (list[str]): The company names extracted by an LLM from the query.
        company_lookup (dict[str, str]): The company lookup table.
        fuzzy_threshold (float): The threshold of fuzzy match. 0.9 by default.

    Returns:
        set(str): A set of matched CIKs. Empty set if no matches are found.
    """
    keys = [_normalize_company_text(name) for name in extracted_company]

    matched_ciks = set()

    for key in keys:
        # 1. Exact match
        if key in company_lookup:
            matched_ciks.add(company_lookup[key])
            continue

        # 2. Fuzzy match
        best_alias = None
        best_score = 0.0

        for alias in company_lookup:
            score = _similarity(key, alias)

            if score > best_score:
                best_alias = alias
                best_score = score

        if best_alias is not None and best_score >= fuzzy_threshold:
            matched_ciks.add(company_lookup[best_alias])

    return matched_ciks


def _filter_indices(
    chunk_ids: list[str],
    inferred_cik: set[str] | None,
    start_year: str | None,
    end_year: str | None,
) -> list[int]:
    """
    Filter candidate indices based on CIK number and dates.

    Args:
        chunk_ids (list[str]): A list of chunk IDs.
        inferred_cik (set[str] | None): A set of CIKs inferred from the query. None if not identified.
        start_year (str | None): The start year of the date specified in the query. None if not specified.
        end_year (str | None): The end year of the date specified in the query. None if not specified.

    Returns:
        list[int]: A list of candidate indices matching the provided information.
    """
    filtered_indices = []
    if start_year is not None:
        start_year = int(start_year)

    if end_year is not None:
        end_year = int(end_year)

    for i, cid in enumerate(chunk_ids):
        cik, date = cid.split("_")[:2]
        year = int(date[:4])

        if inferred_cik is not None and cik not in inferred_cik:
            continue

        if start_year is not None and start_year > year:
            continue

        if end_year is not None and end_year < year:
            continue

        filtered_indices.append(i)

    return filtered_indices


def extract_cadidates_info(
    client: OpenAI,
    query: str,
    model: str,
) -> tuple[list[str] | None, int | None, int | None]:
    """
    Extract structured retrieval filters from a user query.

    Args:
        client (OpenAI): OpenAI client used for structured extraction.
        query (str): User query to analyze.
        model (str): Model used for extraction.

    Returns:
        tuple[list[str] | None, int | None, int | None]: Extracted company names and year range.

    Raises:
        ValueError: If the model response is invalid or missing required fields.
    """
    filter_prompts = PROMPT_HELPER["filter_chunk"]
    messages = build_input_messages(
        filter_prompts["system"],
        filter_prompts["user"],
        query=query,
    )

    raw_response = generate_response(
        client,
        messages,
        model,
    )

    try:
        extracted = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The filter extraction response was not valid JSON. "
            f"Response: {raw_response!r}"
        ) from exc

    if not isinstance(extracted, dict):
        raise ValueError(
            "The filter extraction response must be a JSON object. "
            f"Received: {type(extracted).__name__}"
        )

    required_fields = {"companies", "start_year", "end_year"}
    missing_fields = required_fields - extracted.keys()

    if missing_fields:
        raise ValueError(
            "The filter extraction response is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    companies = extracted["companies"]
    start_year = extracted["start_year"]
    end_year = extracted["end_year"]

    if not isinstance(companies, list) or not all(
        isinstance(company, str) for company in companies
    ):
        raise ValueError("'companies' must be a list of strings.")

    if start_year is not None and not isinstance(start_year, int):
        raise ValueError("'start_year' must be an integer or null.")

    if end_year is not None and not isinstance(end_year, int):
        raise ValueError("'end_year' must be an integer or null.")

    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError(
            f"'start_year' ({start_year}) cannot be later than 'end_year' ({end_year})."
        )

    return companies, start_year, end_year


def get_company_candidate_indices(
    extracted_company: list[str] | None,
    start_year: str | None,
    end_year: str | None,
    company_info: dict[str, dict[str, str]],
    chunk_ids: list[str],
    *,
    fuzzy_threshold: float = 0.9,
) -> np.ndarray | None:
    """
    Extract company names and/or years from the query, resolve company names to CIKs,
    and return matching corpus indices.

    Args:
        extracted_company (list[str] | None): Company names extracted from the query.
        start_year (str | None): The start year of the date specified in the query.
        end_year (str | None): The end year of the date specified in the query.
        company_info (dict[str, dict[str, str]] | None): Company metadata.
        chunk_ids (list[str]): A list of chunk IDs.
        fuzzy_threshold (float): Minimum similarity threshold for fuzzy company name matching.

    Returns:
        candidate_indices (np.ndarray | None):
            - np.ndarray if at least one filter was provided and candidates were found
            - None if no filter was provided or if a provided company could not be matched
    """
    # No valid company name, start year, and end year found in the query
    if not extracted_company and not start_year and not end_year:
        return None

    matched_ciks = None

    if extracted_company:
        company_lookup = _build_company_lookup(company_info)
        matched_ciks = _match_company_to_cik(
            extracted_company,
            company_lookup,
            fuzzy_threshold=fuzzy_threshold,
        )

        if not matched_ciks:
            return None

    candidate_indices = _filter_indices(
        chunk_ids=chunk_ids,
        inferred_cik=matched_ciks,
        start_year=start_year,
        end_year=end_year,
    )

    return np.asarray(candidate_indices, dtype=np.int64)
