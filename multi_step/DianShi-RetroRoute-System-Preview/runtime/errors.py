"""Case-level failures that must not be papered over with search fallbacks."""

from __future__ import annotations


class ServiceExhaustedError(RuntimeError):
    """Planner or single-step failed after retries were exhausted.

    The search must mark the case failed with ``termination_reason`` instead
    of treating the expansion as a dead end (which would silently pollute
    benchmark results).
    """

    def __init__(
        self,
        service: str,
        reason: str,
        *,
        attempts: int = 0,
        cause: Exception | None = None,
        detail: str = "",
    ) -> None:
        self.service = str(service)
        self.reason = str(reason)
        self.attempts = int(attempts)
        self.cause = cause
        self.detail = str(detail or "")
        super().__init__(self.termination_reason)

    @property
    def termination_reason(self) -> str:
        return f"{self.service}_{self.reason}_exhausted"
