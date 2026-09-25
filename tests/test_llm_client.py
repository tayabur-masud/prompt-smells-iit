from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import openai
import pytest

from app.config import LLMConfig
from app.llm_client import LLMClient, LLMError, RateLimiter, is_transient_error


def _response(content: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _status_error(cls: type, status: int, headers: dict[str, str] | None = None) -> Exception:
    # duck-typed response keeps the test independent of the SDK's HTTP library
    request = SimpleNamespace(method="POST", url="https://llm.test/v1/chat/completions")
    response = SimpleNamespace(status_code=status, headers=headers or {}, request=request)
    return cls("error", response=response, body=None)


class FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.kwargs: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.kwargs.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(outcomes: list[Any], attempts: int = 3) -> tuple[LLMClient, FakeCompletions]:
    completions = FakeCompletions(outcomes)
    fake_openai = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = LLMConfig(api_key="test", model="m", temperature=0, max_tokens=50)
    return LLMClient(config, retry_attempts=attempts, client=fake_openai), completions  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda _: None)


def test_sends_configured_parameters() -> None:
    client, completions = make_client([_response('{"smells": []}')])
    assert client.complete("sys", "user") == '{"smells": []}'
    sent = completions.kwargs[0]
    assert sent["model"] == "m" and sent["max_tokens"] == 50 and sent["temperature"] == 0
    assert sent["response_format"] == {"type": "json_object"}
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_retries_rate_limit_then_succeeds() -> None:
    client, completions = make_client(
        [_status_error(openai.RateLimitError, 429, {"retry-after": "0"}), _response("ok")]
    )
    assert client.complete("s", "u") == "ok"
    assert len(completions.kwargs) == 2


def test_gives_up_after_retry_attempts() -> None:
    errors = [_status_error(openai.InternalServerError, 500) for _ in range(3)]
    client, completions = make_client(errors, attempts=3)
    with pytest.raises(LLMError):
        client.complete("s", "u")
    assert len(completions.kwargs) == 3


def test_permanent_error_is_not_retried() -> None:
    client, completions = make_client([_status_error(openai.AuthenticationError, 401)])
    with pytest.raises(LLMError):
        client.complete("s", "u")
    assert len(completions.kwargs) == 1


def test_exhausted_quota_is_fatal_and_not_retried() -> None:
    quota = _status_error(openai.RateLimitError, 429)
    quota.code, quota.type = "insufficient_quota", "insufficient_quota"
    client, completions = make_client([quota])
    with pytest.raises(LLMError) as info:
        client.complete("s", "u")
    assert info.value.fatal and len(completions.kwargs) == 1


def test_google_quota_messages() -> None:
    per_minute = _status_error(openai.RateLimitError, 429)
    per_minute.message = (
        "You exceeded your current quota, please check your plan and billing details. "
        "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
    )
    per_day = _status_error(openai.RateLimitError, 429)
    per_day.message = "Quota exceeded. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    assert is_transient_error(per_minute)
    assert not is_transient_error(per_day)


def test_google_retry_delay_is_read_from_body() -> None:
    from app.llm_client import _retry_after_seconds

    err = _status_error(openai.RateLimitError, 429)
    err.body = [{"error": {"message": "Quota exceeded. Please retry in 26.8s.", "details": []}}]
    assert _retry_after_seconds(err) == pytest.approx(27.8)
    assert _retry_after_seconds(_status_error(openai.RateLimitError, 429, {"retry-after": "3"})) == 3.0


def test_transient_classification() -> None:
    assert is_transient_error(_status_error(openai.RateLimitError, 429))
    assert is_transient_error(_status_error(openai.InternalServerError, 503))
    assert not is_transient_error(_status_error(openai.BadRequestError, 400))
    assert not is_transient_error(ValueError("x"))


def test_api_key_not_exposed_in_config_repr() -> None:
    config = LLMConfig(api_key="sk-super-secret")
    assert "sk-super-secret" not in repr(config) and "sk-super-secret" not in str(config.model_dump())


def test_rate_limiter_disabled_by_default() -> None:
    RateLimiter(None).acquire()
