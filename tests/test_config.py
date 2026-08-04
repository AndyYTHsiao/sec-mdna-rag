import pytest
from dataclasses import dataclass, field
from src.config import Artifact, CorpusConfig, build_dataclass


@dataclass
class Example:
    required: str
    count: int = 3
    tags: list[str] = field(default_factory=list)


def test_build_dataclass_uses_values_defaults_and_default_factories():
    result = build_dataclass(Example, {"required": "value", "extra": "ignored"})

    assert result == Example(required="value")
    assert result.tags is not build_dataclass(Example, {"required": "other"}).tags


def test_build_dataclass_reports_a_missing_required_field():
    with pytest.raises(ValueError, match="Missing required field 'required'"):
        build_dataclass(Example, {})


def test_artifact_hash_changes_when_dependency_configuration_changes():
    small = Artifact("corpus", CorpusConfig(max_tokens=100))
    large = Artifact("corpus", CorpusConfig(max_tokens=200))

    assert (
        Artifact("index", dependencies=(small,)).compute_hash()
        != Artifact("index", dependencies=(large,)).compute_hash()
    )
