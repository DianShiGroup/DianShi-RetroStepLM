"""Shared OpenAI-compatible LLM transport for the Planner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import openai
from openai import OpenAI


@dataclass(frozen=True)
class LlmResponse:
    content: str
    message: Any
    prompt_tokens: int
    completion_tokens: int
    attempts: int
    elapsed_sec: float


class LlmError(RuntimeError):
    """Base error raised after the shared client stops retrying."""

    def __init__(self, message: str, *, attempts: int, cause: Exception | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.cause = cause


class LlmTimeoutError(LlmError):
    """The request or the caller's total time budget expired."""


class LlmCancelledError(LlmError):
    """The caller requested cancellation between attempts."""


def _is_timeout_error(exc: Exception) -> bool:
    timeout_cls = getattr(openai, "APITimeoutError", None)
    return isinstance(exc, TimeoutError) or (
        timeout_cls is not None and isinstance(exc, timeout_cls)
    )


def _is_retryable_error(exc: Exception) -> bool:
    if _is_timeout_error(exc):
        return True

    retryable_classes = tuple(
        cls
        for cls in (
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "RateLimitError", None),
            getattr(openai, "InternalServerError", None),
        )
        if cls is not None
    )
    if retryable_classes and isinstance(exc, retryable_classes):
        return True

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code in (408, 409, 429) or status_code >= 500

    return True


class LlmClient:
    """OpenAI-compatible client with one explicit, observable retry layer."""

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        timeout: float | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.timeout = float(timeout) if timeout is not None else None
        kwargs: dict[str, Any] = {
            "api_key": api_key or "placeholder",
            "max_retries": 0,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        self._client = (client_factory or OpenAI)(**kwargs)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        disable_temperature: bool = False,
        max_tokens: int = 4096,
        extra_body: dict[str, Any] | None = None,
        max_attempts: int = 1,
        retry_delay: float = 1.0,
        retry_backoff: float = 1.0,
        case_timeout: float | None = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> LlmResponse:
        started = time.monotonic()
        max_attempts = max(1, int(max_attempts))
        retry_delay = max(0.0, float(retry_delay))
        retry_backoff = max(1.0, float(retry_backoff))
        deadline = (
            started + float(case_timeout)
            if case_timeout is not None and float(case_timeout) > 0
            else None
        )
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            if should_stop and should_stop():
                raise LlmCancelledError(
                    "LLM request cancelled", attempts=attempt - 1, cause=last_error
                )

            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise LlmTimeoutError(
                    f"LLM case timeout exceeded ({case_timeout}s)",
                    attempts=attempt - 1,
                    cause=last_error,
                )

            request_timeout = self.timeout
            if remaining is not None:
                request_timeout = (
                    min(request_timeout, remaining)
                    if request_timeout is not None
                    else remaining
                )

            try:
                request_client = (
                    self._client.with_options(timeout=request_timeout)
                    if request_timeout is not None
                    else self._client
                )
                request: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": int(max_tokens),
                }
                if not disable_temperature:
                    request["temperature"] = temperature
                if extra_body:
                    request["extra_body"] = extra_body

                response = request_client.chat.completions.create(**request)
                message = response.choices[0].message
                usage = getattr(response, "usage", None)
                return LlmResponse(
                    content=message.content or "",
                    message=message,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    attempts=attempt,
                    elapsed_sec=time.monotonic() - started,
                )
            except Exception as exc:  # noqa: BLE001 - normalized below
                last_error = exc
                if not _is_retryable_error(exc) or attempt >= max_attempts:
                    error_cls = LlmTimeoutError if _is_timeout_error(exc) else LlmError
                    raise error_cls(
                        f"LLM request failed after {attempt} attempt(s): {exc}",
                        attempts=attempt,
                        cause=exc,
                    ) from exc

                delay = retry_delay * (retry_backoff ** (attempt - 1))
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or delay >= remaining:
                        raise LlmTimeoutError(
                            f"LLM case timeout exceeded ({case_timeout}s)",
                            attempts=attempt,
                            cause=exc,
                        ) from exc
                if delay > 0:
                    time.sleep(delay)

        raise LlmError(
            "LLM request failed without an attempt", attempts=0, cause=last_error
        )
