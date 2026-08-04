import json
import pytest
from src.utils import hash_config, load_corpus, tokenize


def test_tokenize_normalizes_case_plurals_and_financial_values():
    assert tokenize("Revenues and CUSTOMER'S costs rose 12.5%.") == [
        "revenue",
        "and",
        "customer's",
        "cost",
        "rose",
        "12.5%",
    ]


def test_load_corpus_skips_blank_lines_and_preserves_order(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"chunk_id": "first", "text": "Alpha"}\n\n'
        '{"chunk_id": "second", "text": "Beta"}\n',
        encoding="utf-8",
    )

    assert load_corpus(path) == (["first", "second"], ["Alpha", "Beta"])


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"chunk_id": "only"}\n', "Invalid row at line 1"),
        (
            '{"chunk_id": "same", "text": "A"}\n{"chunk_id": "same", "text": "B"}\n',
            "Duplicate chunk_ids detected",
        ),
    ],
)
def test_load_corpus_rejects_invalid_rows(tmp_path, contents, message):
    path = tmp_path / "corpus.jsonl"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_corpus(path)


def test_hash_is_stable_across_dictionary_key_order():
    left = {"model": "small", "options": {"b": 2, "a": 1}}
    right = json.loads('{"options": {"a": 1, "b": 2}, "model": "small"}')

    assert hash_config(left) == hash_config(right)
    assert len(hash_config(left)) == 12
