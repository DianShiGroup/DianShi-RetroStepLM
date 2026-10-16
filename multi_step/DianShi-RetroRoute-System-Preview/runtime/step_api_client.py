"""HTTP client for the single-step retro API."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import re
import sqlite3
import threading
import time
from typing import Any

from runtime.errors import ServiceExhaustedError

import httpx
from rdkit import Chem

from runtime.stock import _load_starting_molecules_cached, _resolve_starting_molecules_path
from .single_step.candidate_filters import CandidateFilters
from .single_step.service import SingleStepRAGService

logger = logging.getLogger(__name__)

_SQLITE_STOCK_SUFFIXES = (".sqlite", ".sqlite3", ".db")
_SINGLE_STEP_BACKEND_ALIASES = {
    "dianshi": "dianshi",
    "vllm": "dianshi",
}
_RETRO_CONFIG_KEYS = {
    "api_key",
    "api_key_file",
    "api_url",
    "api_urls",
    "backend",
    "beam_width",
    "disable_temperature",
    "endpoint",
    "endpoints",
    "extra_body",
    "max_tokens",
    "model",
    "num_res",
    "predict_path",
    "prompt_path",
    "retries",
    "server_top_k",
    "timeout",
}


def normalize_single_step_backend(value: str | None) -> str:
    backend = str(value or "").strip().lower()
    try:
        return _SINGLE_STEP_BACKEND_ALIASES[backend]
    except KeyError:
        raise ValueError(
            f"Unsupported single-step backend '{value}'. Expected dianshi"
        ) from None


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_single_step_backend_config(
    config: dict[str, Any], requested_backend: str | None = None
) -> dict[str, Any]:
    """Return a config with the selected single-step backend materialized.

    ``single_step.backends`` may contain complete backend definitions or just
    backend-specific ``retro`` options. The legacy ``single_step.retro`` shape
    remains valid and ``vllm`` is accepted as an alias for ``dianshi``.
    """
    single_step = config.get("single_step", {}) or {}
    if not isinstance(single_step, dict):
        raise ValueError("single_step configuration is invalid")

    legacy_retro = single_step.get("retro", {}) or {}
    if not isinstance(legacy_retro, dict):
        raise ValueError("single_step.retro configuration is invalid")
    configured_raw = legacy_retro.get("backend") or single_step.get("default_backend")
    configured_backend = normalize_single_step_backend(configured_raw or "dianshi")
    selected_backend = normalize_single_step_backend(requested_backend or configured_backend)

    registry = single_step.get("backends")
    if registry is None and isinstance(legacy_retro.get("backends"), dict):
        registry = legacy_retro["backends"]
    if registry is not None and not isinstance(registry, dict):
        raise ValueError("single_step.backends must be a mapping")
    if not registry and requested_backend is not None and selected_backend != configured_backend:
        raise ValueError(
            f"single-step backend '{selected_backend}' is not configured; "
            "add it under single_step.backends"
        )

    if not registry:
        if requested_backend is None and legacy_retro.get("backend") == configured_backend:
            return config
        resolved = copy.deepcopy(config)
        resolved.setdefault("single_step", {}).setdefault("retro", {})[
            "backend"
        ] = selected_backend
        return resolved

    entry = None
    for raw_name, raw_entry in registry.items():
        try:
            if normalize_single_step_backend(raw_name) == selected_backend:
                entry = raw_entry
                break
        except ValueError:
            continue

    if entry is None:
        if requested_backend is None and selected_backend == configured_backend:
            return config
        raise ValueError(
            f"single-step backend '{selected_backend}' is not configured; "
            f"available: {sorted(registry)}"
        )
    if not isinstance(entry, dict):
        raise ValueError(f"single_step.backends.{selected_backend} must be a mapping")

    if isinstance(entry.get("single_step"), dict):
        entry = entry["single_step"]
    base_single = {
        key: value
        for key, value in single_step.items()
        if key not in {"backends", "default_backend"}
    }
    if isinstance(base_single.get("retro"), dict):
        base_single["retro"] = {
            key: value for key, value in base_single["retro"].items() if key != "backends"
        }
    if isinstance(entry.get("retro"), dict):
        resolved_single = _deep_merge_dicts(
            base_single,
            {key: value for key, value in entry.items() if key != "retro"},
        )
        resolved_single["retro"] = _deep_merge_dicts(base_single.get("retro", {}), entry["retro"])
    elif any(key in _RETRO_CONFIG_KEYS for key in entry):
        resolved_single = _deep_merge_dicts(base_single, {"retro": entry})
    else:
        resolved_single = _deep_merge_dicts(base_single, entry)
    resolved_single.setdefault("retro", {})["backend"] = selected_backend

    resolved = copy.deepcopy(config)
    resolved["single_step"] = resolved_single
    return resolved


def _canonicalize_smiles_clear_map(smiles: str) -> str:
    value = str(smiles or "").strip()
    if not value:
        return ""
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _is_sqlite_stock_path(path: str | None) -> bool:
    return bool(path and os.path.splitext(path)[1].lower() in _SQLITE_STOCK_SUFFIXES)


def extract_reasoning_from_message(msg, content: str) -> str:
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        return reasoning

    thinking_match = re.search(
        r"<thinking>\s*(.*?)\s*</thinking>", content, re.DOTALL | re.IGNORECASE
    )
    if thinking_match:
        return thinking_match.group(1).strip()

    tool_match = re.search(r"<tool>", content, re.IGNORECASE)
    if tool_match:
        before_tool = content[: tool_match.start()].strip()
        if before_tool:
            return before_tool

    return ""


class _SQLiteStockLookup:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._cache: dict[str, bool] = {}
        self._count: int | None = None

    def __len__(self) -> int:
        if self._count is None:
            self._count = self._count_rows()
        return self._count

    def _count_rows(self) -> int:
        try:
            with sqlite3.connect(self.path) as conn:
                return int(conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0])
        except Exception as exc:
            logger.error("Failed to count stock sqlite database %s: %s", self.path, exc)
            return 0

    def __contains__(self, value: str) -> bool:
        return self.check_many([value]).get(value, False)

    def check_many(self, smiles_list: list[str]) -> dict[str, bool]:
        unique_smiles = list(dict.fromkeys(s for s in smiles_list if str(s).strip()))
        if not unique_smiles:
            return {}

        to_query = [s for s in unique_smiles if s not in self._cache]
        if to_query:
            try:
                with sqlite3.connect(self.path) as conn:
                    for idx in range(0, len(to_query), 900):
                        chunk = to_query[idx : idx + 900]
                        placeholders = ",".join("?" for _ in chunk)
                        rows = conn.execute(
                            f"SELECT smiles FROM stocks WHERE smiles IN ({placeholders})",
                            chunk,
                        ).fetchall()
                        matched = {str(row[0]) for row in rows}
                        for smile in chunk:
                            self._cache[smile] = smile in matched
            except Exception as exc:
                logger.error("Failed to query stock sqlite database %s: %s", self.path, exc)
                for smile in to_query:
                    self._cache.setdefault(smile, False)

        return {smile: bool(self._cache.get(smile, False)) for smile in unique_smiles}


class StepAPIClient:
    """Single-step adapter with embedded model backends and HTTP fallback."""

    def __init__(
        self,
        *,
        base_url: str,
        starting_molecules: str,
        timeout: int = 120,
        num_res: int | None = None,
        step_service: Any | None = None,
        candidate_filters: CandidateFilters | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.num_res = num_res
        self.step_service = step_service
        self.candidate_filters = candidate_filters
        logger.info(
            "StepAPIClient loading starting molecules | path=%s", starting_molecules
        )
        t0 = time.time()

        resolved_path = _resolve_starting_molecules_path(starting_molecules)
        if resolved_path is not None and _is_sqlite_stock_path(resolved_path):
            self.starting_mols = _SQLiteStockLookup(resolved_path)
            logger.info("StepAPIClient uses stock sqlite backend: %s", resolved_path)
        elif resolved_path is not None:
            self.starting_mols = _load_starting_molecules_cached(resolved_path)
        else:
            self.starting_mols = frozenset()

        if isinstance(self.starting_mols, _SQLiteStockLookup):
            startup_msg = "sqlite backend"
        else:
            startup_msg = f"{len(self.starting_mols)}"

        logger.info(
            "StepAPIClient ready with %s starting molecules | elapsed=%.2fs",
            startup_msg,
            time.time() - t0,
        )

        self._last_raw_result = threading.local()

    @property
    def last_raw_result(self) -> dict[str, Any] | None:
        return getattr(self._last_raw_result, "value", None)

    @last_raw_result.setter
    def last_raw_result(self, value: dict[str, Any] | None) -> None:
        self._last_raw_result.value = value

    def with_candidate_filters(self, candidate_filters: CandidateFilters) -> "StepAPIClient":
        """Return a request-local client that shares the retro service and stock lookup."""
        client = copy.copy(self)
        client.candidate_filters = candidate_filters
        client._last_raw_result = threading.local()
        return client

    def _check_starting_membership(self, smiles_list: list[str]) -> dict[str, bool]:
        if not smiles_list:
            return {}

        unique_smiles = list(dict.fromkeys(smiles_list))

        if isinstance(self.starting_mols, _SQLiteStockLookup):
            return self.starting_mols.check_many(unique_smiles)

        return {smiles: smiles in self.starting_mols for smiles in unique_smiles}

    def _make_availability(self, canonical_smiles: list[str], membership: dict[str, bool]) -> list[list[str]]:
        reaction: list[list[str]] = []
        for smiles in canonical_smiles:
            available = "available" if membership.get(smiles, False) else "unavailable"
            reaction.append([smiles, available])
        return reaction

    def _process_result(
        self,
        molecule: str,
        result: dict[str, Any] | None,
        t0: float,
    ) -> dict[str, Any]:
        if not result:
            logger.warning("Step service returned an empty result | molecule=%.60s", molecule)
            self.last_raw_result = {
                "input_smiles": molecule,
                "ok": False,
                "output_predictions": [],
                "error_message": "empty result",
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
            return {"results": []}

        self.last_raw_result = {
            "input_smiles": molecule,
            "ok": True,
            "output_predictions": [],
            "matched_answer_count": 0,
            "prediction_count": len(result.get("predictions", [])),
            "error_message": None,
            "elapsed_ms": int((time.time() - t0) * 1000),
        }

        converted_results = []
        unique_reactions: list[list[str]] = []
        output_predictions: list[list[str] | None] = [None] * len(
            result.get("predictions", [])
        )
        prepared_predictions: list[tuple[int, list[str], str]] = []
        success_count = 0
        seen = set()

        for prediction_index, prediction in enumerate(result.get("predictions", [])):
            has_answer = prediction.get("has_answer", False)
            if not has_answer:
                content = prediction.get("content", "") or ""
                output_predictions[prediction_index] = [content, "answer_missing"]
                continue

            parsed = prediction.get("parsed", [])
            output_smiles = prediction.get("output_smiles", "") or ""
            if not isinstance(parsed, list) or not parsed:
                parsed = [part.strip() for part in str(output_smiles).split(".") if part.strip()]
            if not parsed:
                output_predictions[prediction_index] = [output_smiles, "answer_empty_parsed"]
                continue

            canonical_reactants: list[str] = []
            invalid = False
            for reactant in parsed:
                canonical = _canonicalize_smiles_clear_map(str(reactant))
                if not canonical:
                    invalid = True
                    break
                canonical_reactants.append(canonical)

            if invalid or not canonical_reactants:
                output_smiles = prediction.get("output_smiles", "") or ""
                output_predictions[prediction_index] = [output_smiles, "smiles_invalid"]
                continue

            dedup_key = tuple(sorted(canonical_reactants))
            if dedup_key in seen:
                output_smiles = prediction.get("output_smiles", "") or ""
                output_predictions[prediction_index] = [output_smiles, "duplicate"]
                continue

            seen.add(dedup_key)
            output_smiles = prediction.get("output_smiles", "") or ""
            prepared_predictions.append(
                (prediction_index, canonical_reactants, output_smiles)
            )

        if (
            self.candidate_filters is not None
            and getattr(self.candidate_filters, "rewrites_candidates", False)
        ):
            kept = self.candidate_filters.filter_list(
                product=molecule,
                candidates=[item[1] for item in prepared_predictions],
            )
            unique_reactions.extend(kept)
            status = "success" if kept else "template_filter_failed"
            for prediction_index, _reactants, output_smiles in prepared_predictions:
                output_predictions[prediction_index] = [output_smiles, status]
            success_count = len(kept)
        else:
            if self.candidate_filters is None:
                filter_results = [None] * len(prepared_predictions)
            elif hasattr(self.candidate_filters, "validate_many"):
                filter_results = self.candidate_filters.validate_many(
                    reactants_batch=[item[1] for item in prepared_predictions],
                    product=molecule,
                )
            else:
                filter_results = [
                    self.candidate_filters.validate(
                        reactants=reactants, product=molecule
                    )
                    for _, reactants, _ in prepared_predictions
                ]

            for (prediction_index, canonical_reactants, output_smiles), filter_result in zip(
                prepared_predictions, filter_results
            ):
                if filter_result is not None and filter_result.status != "PASS":
                    logger.info(
                        "Single-step candidate dropped after %s returned %s: %s",
                        filter_result.gate,
                        filter_result.status,
                        filter_result.detail,
                    )
                    output_predictions[prediction_index] = [
                        output_smiles,
                        f"{filter_result.gate}_failed",
                    ]
                    continue
                unique_reactions.append(canonical_reactants)
                output_predictions[prediction_index] = [output_smiles, "success"]
                success_count += 1

        membership = self._check_starting_membership(
            [smiles for reaction in unique_reactions for smiles in reaction]
        )
        for canonical_reactants in unique_reactions:
            converted_results.append(self._make_availability(canonical_reactants, membership))

        self.last_raw_result["output_predictions"] = [
            output for output in output_predictions if output is not None
        ]
        self.last_raw_result["matched_answer_count"] = success_count
        return {"results": converted_results}

    async def execute_async(self, args: dict[str, Any]) -> dict[str, Any]:
        molecule = str(args.get("molecule", "")).strip()
        if not molecule:
            return {"error": "No molecule provided."}
        if self.step_service is None:
            return self.execute(args)

        self.last_raw_result = None
        t0 = time.time()
        try:
            result = await self.step_service.predict(
                smiles=molecule, num_res=self.num_res
            )
        except asyncio.CancelledError:
            raise
        except ServiceExhaustedError:
            self.last_raw_result = {
                "input_smiles": molecule,
                "ok": False,
                "output_predictions": [],
                "error_message": "single-step service exhausted (see case termination_reason)",
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
            raise
        except Exception as exc:
            logger.exception(
                "Embedded step service failed | molecule=%.60s error=%s",
                molecule,
                exc,
            )
            self.last_raw_result = {
                "input_smiles": molecule,
                "ok": False,
                "output_predictions": [],
                "error_message": str(exc),
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
            return {"results": []}
        return self._process_result(molecule, result, t0)

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        molecule = str(args.get("molecule", "")).strip()
        if not molecule:
            return {"error": "No molecule provided."}

        self.last_raw_result = None
        t0 = time.time()
        if self.step_service is not None:
            try:
                result = asyncio.run(self.execute_async(args))
            except ServiceExhaustedError:
                raise
            except Exception as exc:
                logger.exception(
                    "Embedded step service failed | molecule=%.60s error=%s",
                    molecule,
                    exc,
                )
                self.last_raw_result = {
                    "input_smiles": molecule,
                    "ok": False,
                    "output_predictions": [],
                    "error_message": str(exc),
                    "elapsed_ms": int((time.time() - t0) * 1000),
                }
                return {"results": []}
            return result

        try:
            for trust_env in (False, True):
                with httpx.Client(timeout=self.timeout, trust_env=trust_env) as client:
                    resp = client.post(
                        f"{self.base_url}/retro/predict",
                        json={
                            "smiles": molecule,
                            "num_res": self.num_res,
                        },
                    )
                if (
                    resp.status_code == 502
                    and not trust_env
                    and "response error: EOF" in (resp.text or "").lower()
                ):
                    logger.warning(
                        "Step API direct request got EOF via non-proxy path, retry with env proxies."
                    )
                    continue
                resp.raise_for_status()
                break
        except httpx.HTTPStatusError:
            status_code = resp.status_code
            body_preview = resp.text[:500] if resp.text else ""
            logger.error(
                "Step API HTTP error | molecule=%.60s status=%d body=%.500s",
                molecule,
                status_code,
                body_preview,
            )
            self.last_raw_result = {
                "input_smiles": molecule,
                "ok": False,
                "output_predictions": [],
                "error_message": f"HTTP {status_code}: {body_preview}",
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
            return {"results": []}
        except httpx.TimeoutException:
            logger.error(
                "Step API timeout | molecule=%.60s timeout=%ds",
                molecule,
                self.timeout,
            )
            self.last_raw_result = {
                "input_smiles": molecule,
                "ok": False,
                "output_predictions": [],
                "error_message": f"timeout after {self.timeout}s",
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
            return {"results": []}
        except httpx.ConnectError as e:
            logger.error(
                "Step API connection refused | molecule=%.60s url=%s error=%s",
                molecule,
                self.base_url,
                e,
            )
            self.last_raw_result = {
                "input_smiles": molecule,
                "ok": False,
                "output_predictions": [],
                "error_message": f"connection refused: {e}",
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
            return {"results": []}
        except httpx.RequestError as e:
            logger.error(
                "Step API request failed | molecule=%.60s url=%s error=%s",
                molecule,
                self.base_url,
                e,
            )
            self.last_raw_result = {
                "input_smiles": molecule,
                "ok": False,
                "output_predictions": [],
                "error_message": str(e),
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
            return {"results": []}
        payload = resp.json()
        return self._process_result(molecule, payload.get("result", {}), t0)


def build_embedded_step_api_client(config: dict[str, Any]) -> StepAPIClient:
    """Build the configured in-process single-step client."""
    config = resolve_single_step_backend_config(config)
    single_step_config = config.get("single_step", {}) or {}
    retro_config = single_step_config.get("retro", {}) or {}
    backend = normalize_single_step_backend(retro_config.get("backend", "dianshi"))

    if backend == "dianshi":
        api_urls = retro_config.get("api_url") or retro_config.get("api_urls")
        if not api_urls:
            raise ValueError(
                "single_step.retro.api_url must configure at least one dianshi endpoint"
            )

        rag_enabled = single_step_config.get("rag_enabled", True)
        prompt_path = (
            (retro_config.get("prompt_path_rag") or single_step_config.get("prompt_path_rag"))
            if rag_enabled
            else single_step_config.get("prompt_path_no_rag")
        ) or single_step_config.get("prompt_path_rag")
        chemrag_db_dir = single_step_config.get("chemrag_db_dir") if rag_enabled else None
        rag_selection = str(single_step_config.get("rag_selection", "plain"))

        service = SingleStepRAGService(
            retro_api_url=api_urls,
            retro_model_name=retro_config.get("model", "default"),
            retro_prompt_path=prompt_path,
            retro_num_res=retro_config.get("num_res") or retro_config.get("beam_width", 20),
            chemrag_db_dir=chemrag_db_dir,
            api_key=single_step_config.get("api_key") or None,
            temperature=single_step_config.get("temperature", 0.0),
            max_tokens=single_step_config.get("max_tokens", 1024),
            timeout=single_step_config.get("timeout", 120),
            faiss_candidates=single_step_config.get("faiss_candidates", 200),
            rag_top_k=single_step_config.get("rag_top_k", 5),
            rag_selection=rag_selection,
            atomwise=single_step_config.get("atomwise", False),
        )
        step_service = service
        num_res = service.retro_num_res
        timeout = single_step_config.get("timeout", 120)
        logger.info(
            "Embedded step service ready | backend=dianshi endpoints=%d model=%s rag=%s atomwise=%s",
            len(api_urls) if isinstance(api_urls, list) else 1,
            service.retro_model_name,
            bool(service._chemrag_db),
            service.atomwise,
        )
    else:
        raise ValueError(
            "Unsupported single_step.retro.backend: %s (expected dianshi)" % backend
        )

    return StepAPIClient(
        base_url="",
        starting_molecules=config.get("starting_molecules", ""),
        timeout=timeout,
        num_res=num_res,
        step_service=step_service,
        candidate_filters=CandidateFilters(
            single_step_config,
            starting_molecules_path=config.get("starting_molecules"),
        ),
    )
