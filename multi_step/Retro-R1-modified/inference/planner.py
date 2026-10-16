"""Small OpenAI-compatible client for the Planner model."""

import logging
import re
import time
from typing import Dict, List, Optional

from openai import OpenAI


logger = logging.getLogger(__name__)

_CONTEXT_ERROR_RE = re.compile(
    r"maximum context length is (?P<limit>\d+) tokens.*?"
    r"requested \d+ tokens \((?P<messages>\d+) in the messages, "
    r"(?P<completion>\d+) in the completion\)",
    re.DOTALL,
)


class PlannerError(RuntimeError):
    """Raised when the Planner API cannot produce a response."""


def _fit_context_max_tokens(error: str, requested: int) -> Optional[int]:
    match = _CONTEXT_ERROR_RE.search(error)
    if match is None:
        return None
    available = int(match.group("limit")) - int(match.group("messages")) - 128
    if available < 256:
        return None
    return min(requested - 1, available)


class OpenAICompatiblePlanner:
    """Retrying wrapper around Chat Completions."""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 32768,
        max_retries: int = 5,
        timeout: float = 600.0,
    ) -> None:
        kwargs = {"api_key": api_key, "timeout": timeout}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def complete(self, messages: List[Dict[str, str]]) -> str:
        request_max_tokens = self.max_tokens
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=request_max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise PlannerError("Planner returned no textual content")
                return content
            except Exception as exc:
                last_error = exc
                adjusted = _fit_context_max_tokens(str(exc), request_max_tokens)
                if adjusted is not None:
                    request_max_tokens = adjusted
                    logger.warning(
                        "Planner context is near the model limit; retrying with "
                        "max_tokens=%d",
                        request_max_tokens,
                    )
                    continue
                if attempt + 1 >= self.max_retries:
                    break
                delay = min(2**attempt, 60)
                logger.warning(
                    "Planner API attempt %d/%d failed: %s; retrying in %ds",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise PlannerError(str(last_error)) from last_error
