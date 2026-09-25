from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.conversation_parser import ConversationParser, is_in_sample, normalize_messages
from tests.conftest import msg

TS = datetime(2023, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def record(conversation, conversation_id: str = "c1") -> dict:
    return {"conversation_id": conversation_id, "model": "gpt-4", "timestamp": TS, "conversation": conversation}


def test_extracts_only_user_role() -> None:
    parser = ConversationParser()
    prompts = parser.extract_from_record(record([msg("user", "Hi"), msg("assistant", "Hello")]))
    assert [p.content for p in prompts] == ["Hi"]
    assert parser.stats.user_messages == 1


def test_filters_non_english_messages() -> None:
    parser = ConversationParser()
    conv = [msg("user", "Hello"), msg("user", "Bonjour", "French"), msg("user", "Hola", "Spanish")]
    prompts = parser.extract_from_record(record(conv))
    assert [p.content for p in prompts] == ["Hello"]
    assert parser.stats.user_messages == 3
    assert parser.stats.english_user_messages == 1


def test_message_without_language_is_not_assumed_english() -> None:
    parser = ConversationParser()
    prompts = parser.extract_from_record(record([{"role": "user", "content": "Hello"}]))
    assert prompts == []


def test_multiple_user_messages_keep_metadata() -> None:
    conv = [msg("user", "First"), msg("assistant", "A"), msg("user", "Second")]
    prompts = ConversationParser().extract_from_record(record(conv, "abc"))
    assert [p.content for p in prompts] == ["First", "Second"]
    assert all(p.conversation_id == "abc" and p.model == "gpt-4" for p in prompts)
    assert prompts[0].timestamp == "2023-05-01T12:00:00+00:00"
    assert prompts[0].prompt_id != prompts[1].prompt_id


def test_numpy_array_conversation_is_supported() -> None:
    np = pytest.importorskip("numpy")
    conv = np.array([msg("user", "From numpy")], dtype=object)
    assert [p.content for p in ConversationParser().extract_from_record(record(conv))] == ["From numpy"]


def test_json_string_conversation_is_supported() -> None:
    conv = json.dumps([msg("user", "From JSON")])
    assert [p.content for p in ConversationParser().extract_from_record(record(conv))] == ["From JSON"]


def test_missing_conversation_counts_as_malformed() -> None:
    parser = ConversationParser()
    assert parser.extract_from_record(record(None)) == []
    assert parser.stats.malformed_records == 1
    assert parser.stats.valid_conversations == 0


def test_malformed_conversations_are_skipped() -> None:
    parser = ConversationParser()
    for bad in ["not json", 42, [], ["a string", 7], {"no_messages": True}]:
        assert parser.extract_from_record(record(bad)) == []
    assert parser.extract_from_record({"model": "x", "conversation": [msg("user", "hi")]}) == []
    assert parser.stats.malformed_records == 6
    assert parser.stats.records_read == 6


def test_empty_and_non_string_content_skipped() -> None:
    conv = [msg("user", "   "), {"role": "user", "language": "English", "content": None}, msg("user", "ok")]
    assert [p.content for p in ConversationParser().extract_from_record(record(conv))] == ["ok"]


def test_unicode_preserved() -> None:
    text = "Explain «naïve» café 日本語 🚀"
    assert ConversationParser().extract_from_record(record([msg("user", text)]))[0].content == text


def test_extract_respects_max_prompts() -> None:
    records = [record([msg("user", f"p{i}a"), msg("user", f"p{i}b")], f"c{i}") for i in range(5)]
    prompts = list(ConversationParser().extract(records, max_prompts=3))
    assert [p.content for p in prompts] == ["p0a", "p0b", "p1a"]


def test_normalize_messages_drops_non_dict_items() -> None:
    assert normalize_messages([{"role": "user"}, "junk", None]) == [{"role": "user"}]


def test_sampling_is_deterministic() -> None:
    ids = [f"conv-{i}" for i in range(2000)]
    first = [is_in_sample(i, 0.25) for i in ids]
    assert first == [is_in_sample(i, 0.25) for i in ids]
    assert 0.18 < sum(first) / len(ids) < 0.32
    assert all(is_in_sample(i, None) for i in ids[:10])
