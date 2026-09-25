from __future__ import annotations

import json

import pytest

from app.prompt_analyzer import (
    OTHER_CATEGORY,
    SMELL_CATEGORIES,
    InvalidAnalysisResponse,
    PromptAnalyzer,
    normalize_smell_type,
    parse_analysis_response,
)
from tests.conftest import FakeLLMClient


def test_empty_smell_result() -> None:
    assert parse_analysis_response('{"smells": []}').smells == []


def test_multiple_smell_result() -> None:
    text = json.dumps(
        {
            "smells": [
                {"type": "Vague / Missing Context", "reason": "No detail."},
                {"type": "Ambiguous References", "reason": "'this' is undefined."},
            ]
        }
    )
    result = parse_analysis_response(text)
    assert [s.type for s in result.smells] == ["Vague / Missing Context", "Ambiguous References"]


def test_code_fenced_and_chatty_response_is_repaired() -> None:
    text = 'Here you go:\n```json\n{"smells": [{"type": "Format Ambiguity", "reason": "x"},]}\n```'
    assert parse_analysis_response(text).smells[0].type == "Format Ambiguity"


def test_bare_list_is_accepted() -> None:
    assert parse_analysis_response('[{"type": "Overloaded Prompt", "reason": "r"}]').smells[0].type == (
        "Overloaded Prompt"
    )


@pytest.mark.parametrize(
    "text",
    ["", "not json at all", '{"result": "ok"}', '{"smells": [{"type": "", "reason": "r"}]}',
     '{"smells": [{"reason": "missing type"}]}', '{"smells": "none"}'],
)
def test_invalid_responses_raise(text: str) -> None:
    with pytest.raises(InvalidAnalysisResponse):
        parse_analysis_response(text)


def test_category_normalization() -> None:
    assert normalize_smell_type("vague/missing context") == ("Vague / Missing Context", True)
    assert normalize_smell_type("1. Overloaded Prompt") == ("Overloaded Prompt", True)
    assert normalize_smell_type("Prompt Bloat") == ("Prompt Bloat / Convoluted Prompt", True)
    assert normalize_smell_type("Other") == (OTHER_CATEGORY, True)
    assert normalize_smell_type("Unsafe Request") == (OTHER_CATEGORY, False)


def test_unknown_category_becomes_other_with_label_kept() -> None:
    result = parse_analysis_response('{"smells": [{"type": "Typos", "reason": "Many misspellings."}]}')
    assert result.smells[0].type == OTHER_CATEGORY
    assert result.smells[0].reason == "Typos: Many misspellings."


def test_duplicate_categories_are_collapsed() -> None:
    text = json.dumps({"smells": [{"type": "Format Ambiguity", "reason": "a"},
                                  {"type": "format ambiguity", "reason": "b"}]})
    assert len(parse_analysis_response(text).smells) == 1


def test_analyzer_retries_invalid_response_then_succeeds() -> None:
    replies = iter(["garbage", '{"smells": []}'])
    fake = FakeLLMClient(lambda _: next(replies))
    result = PromptAnalyzer(fake, invalid_response_attempts=3).analyze("Hello")  # type: ignore[arg-type]
    assert result.smells == [] and len(fake.calls) == 2


def test_analyzer_gives_up_after_attempts() -> None:
    fake = FakeLLMClient(lambda _: "still garbage")
    with pytest.raises(InvalidAnalysisResponse):
        PromptAnalyzer(fake, invalid_response_attempts=2).analyze("Hello")  # type: ignore[arg-type]
    assert len(fake.calls) == 2


def test_prompt_is_delimited_and_truncated() -> None:
    fake = FakeLLMClient(lambda _: '{"smells": []}')
    PromptAnalyzer(fake, max_prompt_chars=100).analyze("x" * 500)  # type: ignore[arg-type]
    sent = fake.calls[0]
    assert "<prompt>" in sent and "</prompt>" in sent and "truncated" in sent
    assert sent.count("x") == 100


def test_all_categories_are_named_in_system_prompt() -> None:
    from app.prompt_analyzer import SYSTEM_PROMPT

    assert all(name in SYSTEM_PROMPT for name in SMELL_CATEGORIES)
