import numpy as np
import pytest
from src import chunk_filter


COMPANIES = {
    "320193": {"ticker": "AAPL", "name": "Apple Inc."},
    "1018724": {"ticker": "AMZN", "name": "Amazon.com, Inc."},
}


@pytest.fixture
def company_lookup():
    return chunk_filter._build_company_lookup(COMPANIES)


def test_lookup_supports_name_without_suffix_and_ticker(company_lookup):
    assert company_lookup["apple"] == "320193"
    assert company_lookup["aapl"] == "320193"
    assert company_lookup["amazon com"] == "1018724"


def test_matching_supports_exact_and_fuzzy_names(company_lookup):
    assert chunk_filter._match_company_to_cik(
        ["AAPL", "Amazom"], company_lookup, fuzzy_threshold=0.75
    ) == {"320193", "1018724"}


def test_candidate_indices_combine_company_and_inclusive_year_filters():
    result = chunk_filter.get_company_candidate_indices(
        ["Apple"],
        "2021",
        "2022",
        COMPANIES,
        [
            "320193_2020-09-26_01",
            "320193_2021-09-25_01",
            "320193_2022-09-24_01",
            "1018724_2022-12-31_01",
        ],
    )

    np.testing.assert_array_equal(result, np.array([1, 2], dtype=np.int64))


@pytest.mark.parametrize(
    ("companies", "chunk_ids"),
    [(None, []), (["Unknown"], [])],
)
def test_missing_filters_or_unknown_company_fall_back_to_unfiltered_search(
    companies, chunk_ids
):
    assert (
        chunk_filter.get_company_candidate_indices(
            companies, None, None, COMPANIES, chunk_ids
        )
        is None
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            '{"companies": ["Apple"], "start_year": 2021, "end_year": 2022}',
            (["Apple"], 2021, 2022),
        ),
        (
            '{"companies": [], "start_year": null, "end_year": null}',
            ([], None, None),
        ),
    ],
)
def test_extracts_valid_structured_response(monkeypatch, response, expected):
    monkeypatch.setattr(chunk_filter, "generate_response", lambda *args: response)

    assert chunk_filter.extract_cadidates_info(object(), "query", "model") == expected


def test_rejects_reversed_year_range(monkeypatch):
    response = '{"companies": [], "start_year": 2023, "end_year": 2020}'
    monkeypatch.setattr(chunk_filter, "generate_response", lambda *args: response)

    with pytest.raises(ValueError, match="cannot be later"):
        chunk_filter.extract_cadidates_info(object(), "query", "model")
