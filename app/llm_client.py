"""OpenAI-compatible chat client with retries, backoff and client-side rate limiting."""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from typing import Optional

import openai
from openai import OpenAI
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt

from app.config import LLMConfig

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 60.0


class LLMError(Exception):
    """LLM failure after retries. `fatal` means every later request will fail the same way."""

    def __init__(self, message: str, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


# Not "billing"/"quota" alone: Google's retryable per-minute 429s mention both.
_QUOTA_MARKERS = ("insufficient_quota", "credit_balance", "no credits", "perday")


def is_fatal_error(exc: BaseException) -> bool:
    """Bad credentials, missing permission, unknown model, or exhausted quota/credits."""
    if isinstance(
        exc, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)
    ):
        return True
    if isinstance(exc, openai.RateLimitError):
        details = " ".join(
            str(part) for part in (getattr(exc, "code", ""), getattr(exc, "type", ""), exc.message)
        ).lower()
        return any(marker in details for marker in _QUOTA_MARKERS)
    return False


def is_transient_error(exc: BaseException) -> bool:
    if is_fatal_error(exc):
        return False
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return False


_BODY_DELAY_RE = re.compile(r"retry in ([\d.]+)\s*s|retryDelay['\"]?\s*:\s*['\"]([\d.]+)s", re.I)


def _retry_after_seconds(exc: Optional[BaseException]) -> Optional[float]:
    """Provider-suggested delay: Retry-After header, or Google's retryDelay in the body."""
    if exc is None:
        return None
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = headers.get("retry-after") if headers else None
    if value is not None:
        try:
            return float(value)
        except ValueError:
            pass
    match = _BODY_DELAY_RE.search(f"{getattr(exc, 'body', '')} {exc}")
    if match:
        return float(match.group(1) or match.group(2)) + 1.0
    return None


def backoff_wait(retry_state: RetryCallState) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    hinted = _retry_after_seconds(exc)
    if hinted is not None:
        return min(hinted, MAX_BACKOFF_SECONDS)
    exponential = 2 ** retry_state.attempt_number
    return min(exponential + random.uniform(0, 1), MAX_BACKOFF_SECONDS)


def log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    kind = "API rate limit encountered" if isinstance(exc, openai.RateLimitError) else (
        f"Transient API error ({type(exc).__name__})"
    )
    logger.warning(
        "%s. Retrying in %.1fs (attempt %d)...",
        kind,
        retry_state.upcoming_sleep,
        retry_state.attempt_number + 1,
    )


class RateLimiter:
    """Thread-safe limiter that spaces requests evenly to at most N per minute."""

    def __init__(self, requests_per_minute: Optional[int]) -> None:
        self.interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            slot = max(self._next_slot, now)
            self._next_slot = slot + self.interval
        delay = slot - now
        if delay > 0:
            time.sleep(delay)


class LLMClient:
    def __init__(
        self,
        config: LLMConfig,
        retry_attempts: int = 3,
        requests_per_minute: Optional[int] = None,
        client: Optional[OpenAI] = None,
    ) -> None:
        self.config = config
        self.retry_attempts = retry_attempts
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.client = client or OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            organization=config.organization,
            project=config.project,
            timeout=config.timeout,
            max_retries=0,  # retries are handled here so they are logged and bounded
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        retrying = Retrying(
            retry=retry_if_exception(is_transient_error),
            stop=stop_after_attempt(self.retry_attempts),
            wait=backoff_wait,
            before_sleep=log_retry,
            reraise=True,
        )
        try:
            for attempt in retrying:
                with attempt:
                    return self._request(system_prompt, user_prompt)
        except openai.APIError as exc:
            raise LLMError(
                f"{type(exc).__name__}: {getattr(exc, 'message', exc)}", fatal=is_fatal_error(exc)
            ) from exc
        raise LLMError("LLM request failed without a response")

    def _request(self, system_prompt: str, user_prompt: str) -> str:
        self.rate_limiter.acquire()
        kwargs: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""
