"""Single-step retrosynthesis service: vLLM backend + candidate gates."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
import re
import time
from typing import Any

import httpx

from runtime.errors import ServiceExhaustedError

from .prompt_sections import build_rag_messages, load_prompt_sections
from .smiles_canonical import canonicalize_smiles_clear_map
from .smiles_atomwise import restore_answer_spaces, transform_messages

logger = logging.getLogger(__name__)

ANSWER_RE = re.compile(r"<answer\s*>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _resolve_repo_path(path: str | None) -> str | None:
    if not path:
        return None

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    candidate_paths = [path]
    if not os.path.isabs(path):
        candidate_paths.append(os.path.join(repo_root, path))
        candidate_paths.append(os.path.join(os.getcwd(), path))

    return next(
        (candidate for candidate in candidate_paths if os.path.exists(candidate)), None
    )


@functools.lru_cache(maxsize=4)
def load_chemrag_db_cached(db_dir: str) -> None:
    """ChemRAG retrieval is not part of this release; always run without it."""
    logger.info("ChemRAG retrieval not included; running without RAG | path=%s", db_dir)
    return None


@functools.lru_cache(maxsize=8)
def load_prompt_sections_cached(prompt_path: str) -> dict[str, str]:
    logger.info("Loading RAG prompt | path=%s", prompt_path)
    return load_prompt_sections(prompt_path)


def preload_step_service_resources(
    chemrag_db_dir: str | None,
    retro_prompt_path: str | None,
) -> bool:
    resolved_db_dir = _resolve_repo_path(chemrag_db_dir)
    resolved_retro_prompt_path = _resolve_repo_path(retro_prompt_path)
    if not resolved_db_dir or not resolved_retro_prompt_path:
        logger.warning(
            "Step service preload skipped because db_dir or prompt paths are missing"
        )
        return False
    load_chemrag_db_cached(resolved_db_dir)
    load_prompt_sections_cached(resolved_retro_prompt_path)
    return True


def _extract_answer_tag(content: str) -> str:
    match = ANSWER_RE.search(content or "")
    if not match:
        return ""
    return match.group(1).strip()


def _classify_infra_reason(exc: Exception) -> str:
    """Map a transport/HTTP exception to a short reason token."""
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    status = getattr(exc, "response", None) and getattr(exc.response, "status_code", None)
    if status is not None:
        return f"http_{int(status)}"
    if isinstance(exc, httpx.TransportError):
        return "unreachable"
    return "error"


def _parse_retro_prediction(content: str) -> str:
    answer = _extract_answer_tag(content)
    if not answer:
        return ""
    return answer


def _format_rag_examples(rag_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for idx, item in enumerate(rag_results, start=1):
        examples.append(
            {
                "rank": idx,
                "similarity": item.get("similarity"),
                "reactants": item.get("reactants", ""),
                "products": item.get("products", ""),
                "synthons_smiles": item.get("synthons_smiles", []),
                "leaving_groups": item.get("leaving_groups", []),
                "hash_id": item.get("hash_id"),
            }
        )
    return examples


class SingleStepRAGService:
    """vLLM runner for retro single-step prediction, with optional gates."""

    def __init__(
        self,
        *,
        retro_api_url: str | list[str],
        retro_model_name: str,
        retro_prompt_path: str | None = None,
        retro_num_res: int = 20,
        chemrag_db_dir: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        faiss_candidates: int = 200,
        rag_top_k: int = 5,
        rag_selection: str = "plain",
        atomwise: bool = False,
        api_key: str | None = None,
    ):
        if retro_num_res < 1:
            raise ValueError(f"retro_num_res must be >= 1, got {retro_num_res}")
        self._retro_api_urls: list[str] = (
            retro_api_url if isinstance(retro_api_url, list) else [retro_api_url]
        )
        self.retro_model_name = retro_model_name
        self.retro_num_res = retro_num_res
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "")
        self.faiss_candidates = faiss_candidates
        self.rag_top_k = rag_top_k
        self.rag_selection = str(rag_selection or "plain")
        if self.rag_selection not in ("plain", "unique_template"):
            raise ValueError(
                "rag_selection must be 'plain' or 'unique_template', "
                f"got {self.rag_selection!r}"
            )
        self.atomwise = atomwise

        self._resolved_db_dir = _resolve_repo_path(chemrag_db_dir)
        self._resolved_retro_prompt_path = _resolve_repo_path(retro_prompt_path)

        self._chemrag_db = None  # ChemRAG retrieval is not part of this release.
        self._retro_prompt_sections: dict[str, str] | None = None

        if self._resolved_db_dir:
            try:
                self._chemrag_db = load_chemrag_db_cached(self._resolved_db_dir)
            except Exception:
                logger.warning(
                    "Failed to load ChemRAG DB from %s — continuing without RAG",
                    self._resolved_db_dir,
                    exc_info=True,
                )
                self._chemrag_db = None
        if self._resolved_retro_prompt_path:
            try:
                self._retro_prompt_sections = load_prompt_sections_cached(
                    self._resolved_retro_prompt_path
                )
            except Exception:
                logger.warning(
                    "Failed to load retro prompt from %s",
                    self._resolved_retro_prompt_path,
                    exc_info=True,
                )
                self._retro_prompt_sections = None

    def _build_messages(
        self, smiles: str
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        # ChemRAG retrieval is not part of this release; the plain no-RAG
        # prompt is built with an empty retrieval result.
        db_results: list[dict[str, Any]] = []
        messages = build_rag_messages(
            product_smiles=smiles,
            prompt_sections=self._retro_prompt_sections,
            rag_results=db_results,
            max_examples=self.rag_top_k,
        )
        if self.atomwise:
            messages, transformed_spans = transform_messages(messages)
            logger.info(
                "Atom-wise SMILES formatting enabled | transformed_spans=%d",
                transformed_spans,
            )
        return messages, db_results

    async def _call_vllm(
        self,
        messages: list[dict[str, str]],
        num_res: int | None = None,
    ) -> list[dict[str, Any]]:
        api_url = random.choice(self._retro_api_urls)
        model_name = self.retro_model_name
        num_res = num_res or self.retro_num_res

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "use_beam_search": True,
            "best_of": num_res,
            "n": num_res,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                api_url,
                json=payload,
                headers=headers,
            )
            if resp.is_error:
                error_body = resp.text[:2000] if resp.text else ""
                logger.error(
                    "vLLM API HTTP %d | url=%s model=%s body=%.2000s",
                    resp.status_code,
                    api_url,
                    model_name,
                    error_body,
                )
                resp.raise_for_status()
            return resp.json()

    async def predict_retro(
        self,
        smiles: str,
        num_res: int | None = None,
    ) -> dict[str, Any]:
        canonical_input = canonicalize_smiles_clear_map(smiles)
        if not canonical_input:
            raise ValueError(f"Invalid product SMILES: {smiles}")

        max_retries = 3

        messages, db_results = self._build_messages(canonical_input)

        for attempt in range(1 + max_retries):
            try:
                raw = await self._call_vllm(
                    messages, num_res=num_res
                )
            except Exception as exc:
                if attempt >= max_retries:
                    reason = _classify_infra_reason(exc)
                    logger.exception(
                        "vLLM call failed after %d attempts | smiles=%s | error_type=%s",
                        attempt + 1,
                        smiles[:80],
                        type(exc).__name__,
                    )
                    raise ServiceExhaustedError(
                        "retro",
                        reason,
                        attempts=attempt + 1,
                        cause=exc,
                        detail=f"{type(exc).__name__}: {exc!r}",
                    ) from exc
                delay = min(2**attempt, 10)
                logger.warning(
                    "vLLM call failed; retrying %d/%d in %ds | smiles=%s | error_type=%s | error=%r",
                    attempt + 1,
                    max_retries,
                    delay,
                    smiles[:80],
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(delay)
                continue

            predictions = []
            for idx, choice in enumerate(raw.get("choices", []), start=1):
                content = choice.get("message", {}).get("content", "") or ""
                if self.atomwise:
                    content = restore_answer_spaces(content).text
                answer = _parse_retro_prediction(content)
                if answer:
                    parsed = [part.strip() for part in answer.split(".") if part.strip()]
                    predictions.append(
                        {
                            "rank": idx,
                            "output_smiles": answer,
                            "parsed": parsed,
                            "content": content,
                            "has_answer": True,
                        }
                    )
                else:
                    predictions.append(
                        {
                            "rank": idx,
                            "output_smiles": "",
                            "parsed": [],
                            "content": content,
                            "has_answer": False,
                        }
                    )

            matched = sum(1 for p in predictions if p.get("has_answer", False))
            if matched > 0 or attempt == max_retries:
                break
            delay = min(2**attempt, 10)
            logger.info(
                "Retry %d/%d | smiles=%s... matched=0 | sleeping=%ds",
                attempt + 1, max_retries, smiles[:60], delay,
            )
            await asyncio.sleep(delay)

        return {
            "input_smiles": smiles,
            "canonical_input_smiles": canonical_input,
            "predictions": predictions,
            "rag_examples": _format_rag_examples(db_results),
            "matched_answer_count": matched,
        }

    async def predict(
        self,
        *,
        smiles: str,
        num_res: int | None = None,
    ) -> dict[str, Any]:
        # Check retro result cache first
        canon = canonicalize_smiles_clear_map(smiles)
        effective_num_res = num_res or self.retro_num_res

        try:
            result = await self.predict_retro(
                smiles, num_res=num_res
            )
            result["ok"] = True
            result["error"] = None

            return result
        except ServiceExhaustedError:
            # Infrastructure failure (timeout / unreachable / HTTP error after
            # retries): must fail the whole case, not degrade to an empty
            # expansion. See runtime/errors.py.
            raise
        except Exception as exc:
            error_detail = f"{type(exc).__name__}: {exc!r}"
            logger.exception(
                "Error calling step API | smiles=%s | error=%s",
                smiles[:80],
                error_detail,
            )
            return {
                "input_smiles": smiles,
                "canonical_input_smiles": "",
                "ok": False,
                "predictions": [],
                "rag_examples": [],
                "error": error_detail,
                "cache_hit": False,
            }
