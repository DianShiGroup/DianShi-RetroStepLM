from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from runtime.single_step.candidate_filters import (
    FilterResult,
    ForwardRoundTripFilter,
    LocalTemplateFilter,
)
from runtime.step_api_client import StepAPIClient


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def test_forward_filter_uses_selected_url_and_beam_top3():
    with tempfile.TemporaryDirectory() as temp_dir:
        prompt_path = Path(temp_dir) / "forward.txt"
        prompt_path.write_text(
            "---SYSTEM---\nSystem\n---USER---\n{reactants}\n{rag_examples}\n",
            encoding="utf-8",
        )
        forward_filter = ForwardRoundTripFilter(
            {
                "api_url": ["http://forward-a/v1/chat/completions", "http://forward-b/v1/chat/completions"],
                "api_key": "key",
                "model": "forward-model",
                "prompt_path": str(prompt_path),
                "num_beams": 3,
                "rag_enabled": False,
            },
            chemrag_db_dir=None,
            faiss_candidates=200,
            rag_top_k=5,
        )
        response = _Response(
            {"choices": [{"message": {"content": "<answer>CCO</answer>"}}]}
        )
        with patch(
            "runtime.single_step.candidate_filters.random.choice",
            return_value="http://forward-b/v1/chat/completions",
        ), patch("runtime.single_step.candidate_filters.httpx.post", return_value=response) as post:
            result = forward_filter.validate(reactants=["CC", "O"], product="CCO")

    assert result.status == "PASS"
    assert post.call_args.args[0] == "http://forward-b/v1/chat/completions"
    assert post.call_args.kwargs["json"]["n"] == 3
    assert post.call_args.kwargs["json"]["use_beam_search"] is True


def test_forward_filter_appends_chat_completions_to_v1_base_url():
    forward_filter = ForwardRoundTripFilter(
        {"api_url": ["http://forward-a:10030/v1"]},
        chemrag_db_dir=None,
        faiss_candidates=200,
        rag_top_k=5,
    )

    assert forward_filter.api_urls == ["http://forward-a:10030/v1/chat/completions"]


def test_forward_filter_validates_candidates_concurrently_across_urls():
    with tempfile.TemporaryDirectory() as temp_dir:
        prompt_path = Path(temp_dir) / "forward.txt"
        prompt_path.write_text(
            "---SYSTEM---\nSystem\n---USER---\n{reactants}\n{rag_examples}\n",
            encoding="utf-8",
        )
        forward_filter = ForwardRoundTripFilter(
            {
                "api_url": ["http://forward-a/v1", "http://forward-b/v1", "http://forward-c/v1"],
                "api_key": "key",
                "model": "forward-model",
                "prompt_path": str(prompt_path),
                "rag_enabled": False,
            },
            chemrag_db_dir=None,
            faiss_candidates=200,
            rag_top_k=5,
        )
        response = _Response(
            {"choices": [{"message": {"content": "<answer>CCO</answer>"}}]}
        )
        with patch(
            "runtime.single_step.candidate_filters.httpx.post", return_value=response
        ) as post:
            results = forward_filter.validate_many(
                reactants_batch=[["CC"], ["CCC"], ["CCCC"]], product="CCO"
            )

    assert [result.status for result in results] == ["PASS", "PASS", "PASS"]
    assert {call.args[0] for call in post.call_args_list} == {
        "http://forward-a/v1/chat/completions",
        "http://forward-b/v1/chat/completions",
        "http://forward-c/v1/chat/completions",
    }


def test_template_filter_matches_only_library_templates():
    with tempfile.TemporaryDirectory() as temp_dir:
        template_path = Path(temp_dir) / "templates.txt"
        template_path.write_text("[C:1]>>[C:1].[O:2]\n", encoding="utf-8")
        template_filter = LocalTemplateFilter(
            {
                "template_path": str(template_path),
                "localtemplate_dir": temp_dir,
                "mapper_timeout": 1,
            }
        )
        template_filter._extract_template = lambda _: "[C:1]>>[C:1].[O:2]"  # type: ignore[method-assign]

        result = template_filter.validate(reactants=["C"], product="C")

    assert result.status == "PASS"
    assert template_filter.backend == "localtemplate"


def test_rdchiral_template_filter_matches_jsonl_keys():
    with tempfile.TemporaryDirectory() as temp_dir:
        template_path = Path(temp_dir) / "rdchiral.jsonl"
        template_path.write_text(
            '{"key": "[O:2].[C:1]>>[C:1]", "value": 2}\n',
            encoding="utf-8",
        )
        template_filter = LocalTemplateFilter(
            {
                "backend": "rdchiral",
                "template_path": str(template_path),
                "mapper_timeout": 1,
            }
        )
        template_filter._extract_template = lambda _: "[O:2].[C:1]>>[C:1]"  # type: ignore[method-assign]

        result = template_filter.validate(reactants=["C"], product="C")

    assert result.status == "PASS"
    assert template_filter.backend == "rdchiral"


def test_rdchiral_template_filter_rejects_missing_template():
    with tempfile.TemporaryDirectory() as temp_dir:
        template_path = Path(temp_dir) / "rdchiral.jsonl"
        template_path.write_text('{"key": "[C:1]>>[C:1]", "value": 1}\n', encoding="utf-8")
        template_filter = LocalTemplateFilter(
            {
                "backend": "rdchiral",
                "template_path": str(template_path),
                "mapper_timeout": 1,
            }
        )
        template_filter._extract_template = lambda _: "[C:1]>>[N:1]"  # type: ignore[method-assign]

        result = template_filter.validate(reactants=["C"], product="C")

    assert result.status == "FAIL"


def test_template_filter_timeout_is_fail_for_both_backends():
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    class _TimeoutFuture:
        def result(self, timeout=None):
            raise FuturesTimeoutError()

    for backend, filename, contents in (
        ("rdchiral", "rdchiral.jsonl", '{"key": "[C:1]>>[C:1]", "value": 1}\n'),
        ("localtemplate", "templates.txt", "[C:1]>>[C:1]\n"),
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / filename
            template_path.write_text(contents, encoding="utf-8")
            config = {
                "backend": backend,
                "template_path": str(template_path),
                "mapper_timeout": 1,
            }
            if backend == "localtemplate":
                config["localtemplate_dir"] = temp_dir
            template_filter = LocalTemplateFilter(config)
            with patch(
                "runtime.single_step.candidate_filters._TEMPLATE_EXTRACTION_EXECUTOR.submit",
                return_value=_TimeoutFuture(),
            ):
                result = template_filter.validate(reactants=["C"], product="C")
        assert result.status == "FAIL"
        assert result.detail == "mapper timeout"


def test_template_filter_extraction_error_is_fail_for_both_backends():
    for backend, filename, contents in (
        ("rdchiral", "rdchiral.jsonl", '{"key": "[C:1]>>[C:1]", "value": 1}\n'),
        ("localtemplate", "templates.txt", "[C:1]>>[C:1]\n"),
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / filename
            template_path.write_text(contents, encoding="utf-8")
            config = {
                "backend": backend,
                "template_path": str(template_path),
                "mapper_timeout": 1,
            }
            if backend == "localtemplate":
                config["localtemplate_dir"] = temp_dir
            template_filter = LocalTemplateFilter(config)

            def _boom(_reaction_smiles: str) -> str:
                raise RuntimeError("extract exploded")

            template_filter._extract_template = _boom  # type: ignore[method-assign]
            result = template_filter.validate(reactants=["C"], product="C")
        assert result.status == "FAIL"
        assert result.detail == "RuntimeError"


def test_template_filter_empty_extraction_is_fail():
    with tempfile.TemporaryDirectory() as temp_dir:
        template_path = Path(temp_dir) / "rdchiral.jsonl"
        template_path.write_text('{"key": "[C:1]>>[C:1]", "value": 1}\n', encoding="utf-8")
        template_filter = LocalTemplateFilter(
            {
                "backend": "rdchiral",
                "template_path": str(template_path),
                "mapper_timeout": 1,
            }
        )
        template_filter._extract_template = lambda _: ""  # type: ignore[method-assign]

        result = template_filter.validate(reactants=["C"], product="C")

    assert result.status == "FAIL"


def test_template_filter_rejects_unknown_backend():
    try:
        LocalTemplateFilter({"backend": "desp", "template_path": "x"})
    except ValueError as exc:
        assert "backend" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_step_client_removes_filter_failures_and_unknowns():
    class _Filters:
        def validate_many(self, *, reactants_batch, product):
            assert reactants_batch == [["CO"], ["CC"], ["CCC"]]
            assert product == "CCO"
            return [
                FilterResult("FAIL", "template_filter"),
                FilterResult("UNKNOWN", "forward_filter", "unavailable"),
                FilterResult("PASS", "all"),
            ]

    client = StepAPIClient(
        base_url="",
        starting_molecules="",
        candidate_filters=_Filters(),
    )
    result = client._process_result(
        "CCO",
        {
            "ok": True,
            "predictions": [
                {"has_answer": True, "parsed": ["CO"], "output_smiles": "CO"},
                {"has_answer": True, "parsed": ["CC"], "output_smiles": "CC"},
                {"has_answer": True, "parsed": ["CCC"], "output_smiles": "CCC"},
            ],
        },
        time.time(),
    )

    assert result["results"] == [[["CCC", "unavailable"]]]
    assert client.last_raw_result["output_predictions"] == [
        ["CO", "template_filter_failed"],
        ["CC", "forward_filter_failed"],
        ["CCC", "success"],
    ]


def test_aot_apply_rewrites_and_keeps_template_outputs():
    from runtime.single_step.aot_template_filter import AOTTemplateFilter
    from runtime.single_step.candidate_filters import CandidateFilters

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "rdchiral.jsonl"
        path.write_text(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n',
            encoding="utf-8",
        )
        gate = AOTTemplateFilter(
            {
                "backend": "rdchiral",
                "template_path": str(path),
                "ranking": "none",
                "rule_based": False,
            }
        )
        result = gate.validate(reactants=["C", "O"], product="CO")
        assert result.status == "PASS"
        assert result.detail == "apply-match"
        assert set(result.reactants or ()) == {"C", "O"}

        rewritten = gate.filter_candidates(
            product="CO", candidates=[["C", "O"], ["N"]]
        )
        assert len(rewritten) == 1
        assert set(rewritten[0].reactants or ()) == {"C", "O"}

        filters = CandidateFilters(
            {
                "template_filter": {
                    "enabled": True,
                    "backend": "rdchiral",
                    "match_mode": "apply",
                    "template_path": str(path),
                    "ranking": "none",
                    "rule_based": False,
                }
            }
        )
        assert filters.rewrites_candidates is True
        kept = filters.filter_list(product="CO", candidates=[["N"], ["C", "O"]])
        assert kept == [["C", "O"]] or kept == [["O", "C"]]


def test_aot_apply_loads_desp_idx2template_json():
    from runtime.single_step.aot_template_filter import AOTTemplateFilter

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "idx2template_retro.json"
        path.write_text(
            json.dumps({"0": "[C:1][O:2]>>[C:1].[O:2]"}),
            encoding="utf-8",
        )
        gate = AOTTemplateFilter(
            {
                "backend": "rdchiral",
                "template_path": str(path),
                "ranking": "none",
                "rule_based": False,
            }
        )
        rewritten = gate.filter_candidates(product="CO", candidates=[["N"], ["C", "O"]])
        assert len(rewritten) == 1
        assert set(rewritten[0].reactants or ()) == {"C", "O"}


def test_aot_filter_candidates_parallel_matches_serial(monkeypatch):
    from runtime.single_step.aot_template_filter import AOTTemplateFilter

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "rdchiral.jsonl"
        path.write_text(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n',
            encoding="utf-8",
        )
        gate = AOTTemplateFilter(
            {
                "backend": "rdchiral",
                "template_path": str(path),
                "ranking": "none",
                "rule_based": False,
            }
        )
        cands = [["N"], ["C", "O"], ["CC"]]
        monkeypatch.setenv("AOT_GATE_CAND_WORKERS", "1")
        serial = [
            tuple(sorted(item.reactants or ()))
            for item in gate.filter_candidates(product="CO", candidates=cands)
        ]
        monkeypatch.setenv("AOT_GATE_CAND_WORKERS", "10")
        parallel = [
            tuple(sorted(item.reactants or ()))
            for item in gate.filter_candidates(product="CO", candidates=cands)
        ]
        assert serial == parallel == [("C", "O")]


def test_aot_template_parallel_matches_serial(monkeypatch):
    from runtime.single_step.aot_template_filter import AOTTemplateFilter

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "rdchiral.jsonl"
        path.write_text(
            '{"key": "[C:1][O:2]>>[C:1].[O:2]", "value": 1}\n' * 5,
            encoding="utf-8",
        )
        cfg = {
            "backend": "rdchiral",
            "template_path": str(path),
            "ranking": "none",
            "rule_based": False,
        }
        cands = [["N"], ["C", "O"]]
        monkeypatch.setenv("AOT_GATE_TMPL_WORKERS", "1")
        serial = [
            tuple(sorted(item.reactants or ()))
            for item in AOTTemplateFilter(cfg).filter_candidates(
                product="CO", candidates=cands
            )
        ]
        monkeypatch.setenv("AOT_GATE_TMPL_WORKERS", "4")
        parallel = [
            tuple(sorted(item.reactants or ()))
            for item in AOTTemplateFilter(cfg).filter_candidates(
                product="CO", candidates=cands
            )
        ]
        assert serial == parallel == [("C", "O")]


def test_step_client_uses_aot_rewritten_list():
    class _Filters:
        rewrites_candidates = True

        def filter_list(self, *, product, candidates):
            assert product == "CO"
            assert candidates == [["N"], ["C", "O"]]
            return [["C", "O"]]

    client = StepAPIClient(
        base_url="",
        starting_molecules="",
        candidate_filters=_Filters(),
    )
    result = client._process_result(
        "CO",
        {
            "ok": True,
            "predictions": [
                {"has_answer": True, "parsed": ["N"], "output_smiles": "N"},
                {"has_answer": True, "parsed": ["C", "O"], "output_smiles": "C.O"},
            ],
        },
        time.time(),
    )

    assert result["results"] == [[["C", "unavailable"], ["O", "unavailable"]]]
    assert client.last_raw_result["matched_answer_count"] == 1
