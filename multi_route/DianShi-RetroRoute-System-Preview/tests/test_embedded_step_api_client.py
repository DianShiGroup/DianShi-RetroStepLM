from __future__ import annotations

from runtime.step_api_client import (
    StepAPIClient,
    build_embedded_step_api_client,
)


class _FakeEmbeddedService:
    async def predict(self, *, smiles: str, num_res: int | None = None) -> dict:
        return {
            "input_smiles": smiles,
            "ok": True,
            "predictions": [
                {
                    "rank": 1,
                    "output_smiles": "CCO",
                    "parsed": ["CCO"],
                    "content": "<answer>CCO</answer>",
                    "has_answer": True,
                },
                {
                    "rank": 2,
                    "output_smiles": "",
                    "parsed": [],
                    "content": "no answer",
                    "has_answer": False,
                },
            ],
        }


def test_embedded_step_client_preserves_prediction_contract():
    client = StepAPIClient(
        base_url="",
        starting_molecules="",
        num_res=2,
        step_service=_FakeEmbeddedService(),
    )

    result = client.execute({"molecule": "CCO"})

    assert result == {"results": [[["CCO", "unavailable"]]]}
    assert client.last_raw_result["prediction_count"] == 2
    assert client.last_raw_result["matched_answer_count"] == 1
    assert client.last_raw_result["output_predictions"][1][1] == "answer_missing"


def test_process_result_accepts_retroagent_cache_payload():
    client = StepAPIClient(base_url="", starting_molecules="")
    result = client._process_result(
        "CCO",
        {
            "predictions": [
                {
                    "rank": 1,
                    "has_answer": True,
                    "output_smiles": "CC.O",
                }
            ]
        },
        0.0,
    )
    assert result["results"] == [[["CC", "unavailable"], ["O", "unavailable"]]]
    assert client.last_raw_result["matched_answer_count"] == 1


def test_embedded_step_client_uses_single_step_config():
    client = build_embedded_step_api_client(
        {
            "starting_molecules": "",
            "single_step": {
                "rag_enabled": False,
                "prompt_path_no_rag": (
                    "runtime/single_step/prompts/"
                    "single_step_prompt_no_rag.txt"
                ),
                "retro": {
                    "api_url": "http://vllm.test/v1/chat/completions",
                    "model": "test-model",
                    "num_res": 3,
                },
            },
        }
    )

    assert client.base_url == ""
    assert client.num_res == 3
    assert client.step_service.retro_model_name == "test-model"


def test_async_single_step_timeout_cancels_inflight_work():
    import asyncio

    import pytest

    from runtime.aot.aot_worker import TaskRequest, run_worker
    from runtime.errors import ServiceExhaustedError

    cancelled = {"n": 0}

    async def slow(_smiles):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled["n"] += 1
            raise
        return [["C"]]

    with pytest.raises(ServiceExhaustedError) as exc_info:
        asyncio.run(
            run_worker(
                TaskRequest(
                    task_id="t0",
                    node_id="0",
                    molecule_smiles="CCO",
                    depth=0,
                ),
                slow,
                timeout=0.2,
            )
        )
    assert exc_info.value.termination_reason == "retro_timeout_exhausted"
    assert cancelled["n"] == 1
