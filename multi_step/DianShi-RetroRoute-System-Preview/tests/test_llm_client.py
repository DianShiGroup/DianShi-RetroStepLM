from __future__ import annotations

from types import SimpleNamespace

from runtime.planner_models import resolve_llm_config
from runtime.shared.llm_client import LlmClient, LlmError


class _FakeCompletions:
    def __init__(self, outcomes):
        self._outcomes = outcomes
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeOpenAIClient:
    def __init__(self, outcomes):
        self.chat = SimpleNamespace(completions=_FakeCompletions(outcomes))

    def with_options(self, *, timeout):
        return self


def _response(content="ok", prompt_tokens=3, completion_tokens=2):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def test_client_retries_and_returns_usage():
    fake = _FakeOpenAIClient([TimeoutError("slow"), _response("done")])
    client = LlmClient(client_factory=lambda **kwargs: fake)

    result = client.chat(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
        max_tokens=123,
        extra_body={"thinking": {"type": "enabled"}},
        max_attempts=2,
        retry_delay=0,
    )

    assert result.content == "done"
    assert result.attempts == 2
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2
    assert fake.chat.completions.requests[-1]["temperature"] == 0.2
    assert fake.chat.completions.requests[-1]["extra_body"] == {
        "thinking": {"type": "enabled"}
    }


def test_client_omits_temperature_when_explicitly_disabled():
    fake = _FakeOpenAIClient([_response()])
    client = LlmClient(client_factory=lambda **kwargs: fake)

    client.chat(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        disable_temperature=True,
    )

    assert "temperature" not in fake.chat.completions.requests[-1]


def test_client_does_not_retry_non_retryable_http_error():
    error = RuntimeError("unauthorized")
    error.status_code = 401
    fake = _FakeOpenAIClient([error, _response()])
    client = LlmClient(client_factory=lambda **kwargs: fake)

    try:
        client.chat(
            model="model",
            messages=[],
            max_attempts=3,
            retry_delay=0,
        )
    except LlmError as exc:
        assert exc.attempts == 1
    else:
        raise AssertionError("Expected non-retryable error")
    assert len(fake.chat.completions.requests) == 1


def test_provider_settings_resolve_for_planner():
    config = {
        "planner": {"default_model": "p/chosen"},
        "llm_config": {
            "providers": {
                "p": {
                    "base_url": "http://provider/v1",
                    "api_key": "key",
                    "temperature": 0.0,
                    "max_tokens": 1000,
                    "timeout": 300,
                    "llms": {
                        "chosen": {
                            "model_name": "api-model",
                            "disable_temperature": True,
                            "max_tokens": 2000,
                            "extra_body": {"thinking": {"type": "enabled"}},
                        }
                    },
                }
            }
        },
    }

    resolved = resolve_llm_config(config, "p/chosen")
    assert resolved["model"] == "api-model"
    assert resolved["disable_temperature"] is True
    assert resolved["max_tokens"] == 2000
    assert resolved["timeout"] == 300
    assert resolved["extra_body"] == {"thinking": {"type": "enabled"}}


if __name__ == "__main__":
    test_client_retries_and_returns_usage()
    test_client_omits_temperature_when_explicitly_disabled()
    test_client_does_not_retry_non_retryable_http_error()
    test_provider_settings_resolve_for_planner()
