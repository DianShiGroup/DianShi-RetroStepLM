"""Optional forward and reaction-template filters for retro single-step candidates."""

from __future__ import annotations

import functools
import json
import logging
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

import httpx

from .prompt_sections import build_forward_messages
from .service import (
    _extract_answer_tag,
    load_chemrag_db_cached,
    load_prompt_sections_cached,
)
from .smiles_canonical import (
    canonicalize_reactants,
    canonicalize_smiles_clear_map,
)

logger = logging.getLogger(__name__)

_TEMPLATE_BACKENDS = {"localtemplate", "rdchiral"}
_TEMPLATE_EXTRACTION_LOCK = threading.Lock()
_TEMPLATE_EXTRACTION_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="template-extract"
)


@dataclass(frozen=True)
class FilterResult:
    status: str
    gate: str
    detail: str = ""
    reactants: tuple[str, ...] | None = None
    rank_score: float | None = None


def _resolve_repo_path(path: str | None) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path if os.path.exists(path) else None

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    for candidate in (path, os.path.join(repo_root, path), os.path.join(os.getcwd(), path)):
        if os.path.exists(candidate):
            return candidate
    return None


def _as_chat_completion_url_list(value: Any) -> list[str]:
    def _normalize(item: Any) -> str:
        url = str(item).strip().rstrip("/")
        if not url:
            return ""
        if url.endswith("/chat/completions"):
            return url
        return f"{url}/chat/completions"

    if isinstance(value, str):
        url = _normalize(value)
        return [url] if url else []
    if not isinstance(value, list):
        return []
    return [url for item in value if (url := _normalize(item))]


def _canonicalize_template(template_smarts: str) -> str:
    if not template_smarts or ">>" not in template_smarts:
        return ""
    reactants, products = template_smarts.split(">>", 1)

    def _sort_side(side: str) -> str:
        return ".".join(sorted(part for part in side.split(".") if part.strip()))

    return f"{_sort_side(reactants)}>>{_sort_side(products)}"


@functools.lru_cache(maxsize=4)
def _load_template_library(template_path: str, fmt: str) -> frozenset[str]:
    templates: set[str] = set()
    with open(template_path, encoding="utf-8") as template_file:
        if fmt == "jsonl":
            for line in template_file:
                if not line.strip():
                    continue
                key = json.loads(line).get("key")
                if isinstance(key, str) and (normalized := _canonicalize_template(key)):
                    templates.add(normalized)
        else:
            for line in template_file:
                if normalized := _canonicalize_template(line.strip()):
                    templates.add(normalized)
    if not templates:
        raise ValueError(f"template library is empty: {template_path}")
    return frozenset(templates)


@functools.lru_cache(maxsize=1)
def _load_rxnmapper():
    from rxnmapper import RXNMapper

    return RXNMapper()


@functools.lru_cache(maxsize=4)
def _load_local_template_extractor(localtemplate_dir: str):
    if not os.path.isdir(os.path.join(localtemplate_dir, "LocalTemplate")):
        raise FileNotFoundError(
            f"LocalTemplate package is not present under {localtemplate_dir}"
        )
    if localtemplate_dir not in sys.path:
        sys.path.insert(0, localtemplate_dir)

    from LocalTemplate.template_extractor import (  # type: ignore[import-not-found]
        default_setting,
        extract_from_reaction,
    )

    return extract_from_reaction, default_setting


@functools.lru_cache(maxsize=1)
def _load_rdchiral_extractor():
    from rdchiral import template_extractor as rdchiral_template_extractor

    return rdchiral_template_extractor.extract_from_reaction


class ForwardRoundTripFilter:
    """Validate a retro candidate by predicting its product with a forward model."""

    gate_name = "forward_filter"

    def __init__(
        self,
        config: dict[str, Any],
        *,
        chemrag_db_dir: str | None,
        faiss_candidates: int,
        rag_top_k: int,
    ) -> None:
        self.api_urls = _as_chat_completion_url_list(config.get("api_url"))
        self.api_key = str(config.get("api_key") or "")
        self.model = str(config.get("model") or "").strip()
        self.prompt_path = _resolve_repo_path(config.get("prompt_path"))
        self.num_beams = max(1, int(config.get("num_beams", 3)))
        self.max_tokens = max(1, int(config.get("max_tokens", 2048)))
        self.timeout = float(config.get("timeout", 300))
        self.rag_enabled = bool(config.get("rag_enabled", True))
        self.chemrag_db_dir = _resolve_repo_path(chemrag_db_dir)
        self.faiss_candidates = max(1, int(faiss_candidates))
        self.rag_top_k = max(0, int(rag_top_k))

    def _build_messages(self, reactants: str) -> list[dict[str, str]]:
        if not self.prompt_path:
            raise FileNotFoundError("forward_filter.prompt_path does not exist")

        rag_results: list[dict[str, Any]] = []
        if self.rag_enabled and self.chemrag_db_dir and self.rag_top_k:
            try:
                chemrag_db = load_chemrag_db_cached(self.chemrag_db_dir)
                rag_results = chemrag_db.query_top_forward(
                    reactants,
                    faiss_candidates=self.faiss_candidates,
                    top_k=self.rag_top_k,
                )
            except Exception:
                logger.warning("Forward ChemRAG query failed; continuing without examples", exc_info=True)

        return build_forward_messages(
            reactants=reactants,
            prompt_sections=load_prompt_sections_cached(self.prompt_path),
            rag_results=rag_results,
            max_examples=self.rag_top_k,
        )

    def _validate_with_url(
        self, *, reactants: list[str], product: str, api_url: str | None
    ) -> FilterResult:
        canonical_reactants = canonicalize_reactants(".".join(reactants))
        canonical_product = canonicalize_smiles_clear_map(product)
        if not canonical_reactants or not canonical_product:
            return FilterResult("UNKNOWN", self.gate_name, "invalid canonical input")
        if not api_url or not self.model:
            return FilterResult("UNKNOWN", self.gate_name, "forward endpoint is not configured")

        try:
            response = httpx.post(
                api_url,
                json={
                    "model": self.model,
                    "messages": self._build_messages(canonical_reactants),
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "use_beam_search": True,
                    "best_of": self.num_beams,
                    "n": self.num_beams,
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                timeout=self.timeout,
                trust_env=False,
            )
            response.raise_for_status()
            choices = response.json().get("choices", [])
        except Exception as exc:
            logger.warning("Forward filter request failed; dropping candidate: %r", exc)
            return FilterResult("FAIL", self.gate_name, type(exc).__name__)

        predicted_products = []
        for choice in choices if isinstance(choices, list) else []:
            content = str((choice.get("message") or {}).get("content") or "")
            answer = _extract_answer_tag(content)
            canonical_answer = canonicalize_smiles_clear_map(answer)
            if canonical_answer:
                predicted_products.append(canonical_answer)

        if not predicted_products:
            return FilterResult("FAIL", self.gate_name, "no valid answer")
        if canonical_product in predicted_products:
            return FilterResult("PASS", self.gate_name)
        return FilterResult("FAIL", self.gate_name, "product is absent from forward top-k")

    def validate(self, *, reactants: list[str], product: str) -> FilterResult:
        api_url = random.choice(self.api_urls) if self.api_urls else None
        return self._validate_with_url(
            reactants=reactants, product=product, api_url=api_url
        )

    def validate_many(
        self, *, reactants_batch: list[list[str]], product: str
    ) -> list[FilterResult]:
        if len(reactants_batch) < 2:
            return [self.validate(reactants=reactants, product=product) for reactants in reactants_batch]
        if not self.api_urls or not self.model:
            return [
                self._validate_with_url(reactants=reactants, product=product, api_url=None)
                for reactants in reactants_batch
            ]

        with ThreadPoolExecutor(
            max_workers=min(len(reactants_batch), len(self.api_urls)),
            thread_name_prefix="forward-filter",
        ) as executor:
            futures = [
                executor.submit(
                    self._validate_with_url,
                    reactants=reactants,
                    product=product,
                    api_url=self.api_urls[index % len(self.api_urls)],
                )
                for index, reactants in enumerate(reactants_batch)
            ]
            return [future.result() for future in futures]


class LocalTemplateFilter:
    """Validate a reaction against a WanJ 0314 template library."""

    gate_name = "template_filter"

    def __init__(self, config: dict[str, Any]) -> None:
        backend = str(config.get("backend") or "localtemplate").strip().lower()
        if backend not in _TEMPLATE_BACKENDS:
            raise ValueError(
                f"template_filter.backend must be one of {sorted(_TEMPLATE_BACKENDS)}, got {backend!r}"
            )
        self.backend = backend
        self.template_path = _resolve_repo_path(config.get("template_path"))
        self.localtemplate_dir = _resolve_repo_path(config.get("localtemplate_dir"))
        self.mapper_timeout = float(config.get("mapper_timeout", 60))
        self._library_fmt = "jsonl" if backend == "rdchiral" else "txt"

    def _map_reaction(self, reaction_smiles: str) -> tuple[str, str]:
        mapper = _load_rxnmapper()
        mapped_reaction = mapper.get_attention_guided_atom_maps(
            [reaction_smiles], canonicalize_rxns=False
        )[0]["mapped_rxn"]
        reactants, products = mapped_reaction.split(">>", 1)
        return reactants, products

    def _extract_localtemplate(self, reaction_smiles: str) -> str:
        reactants, products = self._map_reaction(reaction_smiles)
        extract_from_reaction, default_setting = _load_local_template_extractor(
            self.localtemplate_dir or ""
        )
        setting = default_setting.copy()
        setting.update({"use_stereo": True, "use_symbol": True, "retro": True})
        template_info = extract_from_reaction(
            {"_id": 0, "reactants": reactants, "products": products}, setting
        )
        return _canonicalize_template(str((template_info or {}).get("reaction_smarts") or ""))

    def _extract_rdchiral(self, reaction_smiles: str) -> str:
        reactants, products = self._map_reaction(reaction_smiles)
        extract_from_reaction = _load_rdchiral_extractor()
        template_info = extract_from_reaction(
            {"_id": 0, "reactants": reactants, "products": products}
        )
        return _canonicalize_template(str((template_info or {}).get("reaction_smarts") or ""))

    def _extract_template(self, reaction_smiles: str) -> str:
        if self.backend == "rdchiral":
            return self._extract_rdchiral(reaction_smiles)
        return self._extract_localtemplate(reaction_smiles)

    def validate(self, *, reactants: list[str], product: str) -> FilterResult:
        canonical_reactants = canonicalize_reactants(".".join(reactants))
        canonical_product = canonicalize_smiles_clear_map(product)
        if not canonical_reactants or not canonical_product:
            return FilterResult("UNKNOWN", self.gate_name, "invalid canonical input")
        if not self.template_path:
            return FilterResult("UNKNOWN", self.gate_name, "template paths are not configured")
        if self.backend == "localtemplate" and not self.localtemplate_dir:
            return FilterResult("UNKNOWN", self.gate_name, "template paths are not configured")

        try:
            templates = _load_template_library(self.template_path, self._library_fmt)
            with _TEMPLATE_EXTRACTION_LOCK:
                future = _TEMPLATE_EXTRACTION_EXECUTOR.submit(
                    self._extract_template, f"{canonical_reactants}>>{canonical_product}"
                )
                template = _canonicalize_template(
                    future.result(timeout=self.mapper_timeout)
                )
        except FuturesTimeoutError:
            logger.warning("%s extraction timed out; dropping candidate", self.backend)
            return FilterResult("FAIL", self.gate_name, "mapper timeout")
        except Exception as exc:
            logger.warning("%s extraction failed; dropping candidate: %r", self.backend, exc)
            return FilterResult("FAIL", self.gate_name, type(exc).__name__)

        if not template:
            return FilterResult("FAIL", self.gate_name, "template is absent from library")
        if template in templates:
            return FilterResult("PASS", self.gate_name)
        return FilterResult("FAIL", self.gate_name, "template is absent from library")


def _parse_match_mode(raw: Any) -> str:
    mode = str(raw or "exact").strip().lower()
    if mode in {"apply", "apply-match", "aot"}:
        return "apply"
    if mode in {"exact", "exact-match", "key", ""}:
        return "exact"
    raise ValueError(
        "template_filter.match_mode must be 'exact' or 'apply', got %r" % raw
    )


class CandidateFilters:
    """Run enabled single-step candidate filters in documented order."""

    def __init__(
        self,
        single_step_config: dict[str, Any],
        starting_molecules_path: str | None = None,
    ) -> None:
        forward_config = single_step_config.get("forward_filter") or {}
        template_config = single_step_config.get("template_filter") or {}

        self.forward_filter = None
        if forward_config.get("enabled"):
            self.forward_filter = ForwardRoundTripFilter(
                forward_config,
                chemrag_db_dir=single_step_config.get("chemrag_db_dir"),
                faiss_candidates=single_step_config.get("faiss_candidates", 200),
                rag_top_k=single_step_config.get("rag_top_k", 5),
            )

        self.template_filter = None
        self.aot_filter = None
        if template_config.get("enabled"):
            if _parse_match_mode(template_config.get("match_mode")) == "apply":
                from .aot_template_filter import AOTTemplateFilter, load_aot_starting_mols

                stock_path = template_config.get("starting_molecules") or starting_molecules_path
                self.aot_filter = AOTTemplateFilter(
                    template_config,
                    starting_mols=load_aot_starting_mols(stock_path),
                )
            else:
                self.template_filter = LocalTemplateFilter(template_config)

    @property
    def rewrites_candidates(self) -> bool:
        return self.aot_filter is not None

    def validate(self, *, reactants: list[str], product: str) -> FilterResult:
        for candidate_filter in (self.forward_filter, self.aot_filter, self.template_filter):
            if candidate_filter is None:
                continue
            result = candidate_filter.validate(reactants=reactants, product=product)
            if result.status != "PASS":
                return result
        return FilterResult("PASS", "all")

    def validate_many(
        self, *, reactants_batch: list[list[str]], product: str
    ) -> list[FilterResult]:
        if self.forward_filter is None:
            results = [FilterResult("PASS", "all") for _ in reactants_batch]
        else:
            results = self.forward_filter.validate_many(
                reactants_batch=reactants_batch, product=product
            )

        template_filter = self.aot_filter or self.template_filter
        if template_filter is None:
            return results
        return [
            template_filter.validate(reactants=reactants, product=product)
            if result.status == "PASS"
            else result
            for reactants, result in zip(reactants_batch, results)
        ]

    def filter_list(
        self, *, product: str, candidates: list[list[str]]
    ) -> list[list[str]]:
        surviving = list(candidates)
        if self.forward_filter is not None and surviving:
            results = self.forward_filter.validate_many(
                reactants_batch=surviving, product=product
            )
            surviving = [
                reactants
                for reactants, result in zip(surviving, results)
                if result.status == "PASS"
            ]
        if self.aot_filter is not None:
            kept = self.aot_filter.filter_candidates(
                product=product, candidates=surviving
            )
            return [list(item.reactants) for item in kept if item.reactants]
        if self.template_filter is not None:
            return [
                reactants
                for reactants in surviving
                if self.template_filter.validate(
                    reactants=reactants, product=product
                ).status
                == "PASS"
            ]
        return surviving
