"""WanJ rdchiral template gate for RetroAgent DianShi-Direct single-step candidates.

Default match_mode is ``apply`` (AOT* / LLM-Syn-Planner style):

1. PatternFP prefilter, substructure-match both sides of a library SMARTS,
   then ``rdchiralRun`` matching templates (up to ``apply_max_templates``,
   default 256) and rewrite reactants. With ``ranking: aot``, score those
   splits by stock/SCScore without unsolved penalty (cross-template).
   ``ranking: none`` keeps first-hit. If none apply, fall through to blurry.
2. If that misses, blurry: keep product Tversky>=1 templates, rerank that
   pool by LLM-reactant vs template-reactant PatternFP Tversky, take
   ``blurry_top_n`` (default 1000), then rewrite from ``rdchiralRun``.

With ``ranking: aot`` (default), blurry also scores across templates with
unsolved penalty (default 0.1). Auto-fix remains enabled in both modes.

A generated split whose reactants overlap the product is rejected
(AOT* ``updated_set_valid``).

``match_mode: exact`` keeps the tagged extract-then-key-equality gate.
``blurry: false`` keeps step 1 only.
"""

from __future__ import annotations

import functools
import gc
import itertools
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from collections.abc import Container
from typing import Any, Optional

from rdkit import Chem, DataStructs, RDLogger

from inference.gate.aot_reactant_ranker import AOTReactantRanker
from inference.support.smiles_canonical import canonicalize_reactants, canonicalize_smiles_clear_map

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

# The rdchiral template library is never bundled with this repository. It is
# resolved as: explicit config value > environment variable, and a fresh clone
# gets an error naming the variable to set.
#   WANJ_RDCHIRAL_TEMPLATE_JSONL  rdchiral template library (template_path)
_WANJ_RDCHIRAL_JSONL = os.environ.get("WANJ_RDCHIRAL_TEMPLATE_JSONL", "").strip()

_TEMPLATE_LOCK = threading.Lock()
_TEMPLATE_EXECUTOR = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="ra-rdchiral-gate"
)

_APPLY_MODES = {"apply", "apply-match", "substructure"}
_EXACT_MODES = {"exact", "exact-match", "key"}


@dataclass(frozen=True)
class FilterResult:
    status: str
    gate: str
    detail: str = ""
    reactants: Optional[tuple[str, ...]] = None
    rank_score: Optional[float] = None


@dataclass(frozen=True)
class ApplyLibrary:
    smarts: tuple[str, ...]
    product_fps: tuple[Any, ...]
    reactant_fps: tuple[Any, ...]


def _canonicalize_template(template_smarts: str) -> str:
    if not template_smarts or ">>" not in template_smarts:
        return ""
    reactants, products = template_smarts.split(">>", 1)

    def _sort_side(side: str) -> str:
        return ".".join(sorted(part for part in side.split(".") if part.strip()))

    return f"{_sort_side(reactants)}>>{_sort_side(products)}"


def _resolve_path(path: str | None) -> str | None:
    if not path:
        return None
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in (path, os.path.join(repo_root, path), os.path.join(os.getcwd(), path)):
        if os.path.exists(candidate):
            return candidate
    return None


def _parse_mode(raw: Any) -> str:
    mode = str(raw or "apply").strip().lower()
    if mode in _APPLY_MODES:
        return "apply"
    if mode in _EXACT_MODES:
        return "exact"
    raise ValueError(
        "template_filter.match_mode must be 'apply' or 'exact', got %r" % raw
    )


def _as_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(raw: Any, default: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _worker_count(configured: Any, env_name: str, default: int, limit: int) -> int:
    raw = configured if configured is not None else os.environ.get(env_name)
    n = _positive_int(raw, default)
    if limit < 1:
        return 1
    return max(1, min(n, limit))


def _canonical_reactant_tuple(reactants: list[str]) -> tuple[str, ...] | None:
    canonical: list[str] = []
    for smiles in reactants:
        cleaned = canonicalize_smiles_clear_map(smiles)
        if not cleaned:
            return None
        canonical.append(cleaned)
    return tuple(canonical) if canonical else None


def _split_smarts_mols(side: str) -> tuple[Any, ...] | None:
    mols = []
    for part in side.split("."):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmarts(part)
        if mol is None:
            return None
        mols.append(mol)
    return tuple(mols) if mols else None


def _pattern_fp_from_mols(mols: tuple[Any, ...]) -> Any:
    combined = None
    for mol in mols:
        if mol is None:
            continue
        fp = Chem.PatternFingerprint(mol)
        combined = fp if combined is None else (combined | fp)
    return combined if combined is not None else DataStructs.ExplicitBitVect(2048)


def _pattern_fp_from_smarts_side(side: str) -> Any:
    combined = None
    for part in side.split("."):
        mol = Chem.MolFromSmarts(part)
        if mol is None:
            continue
        fp = Chem.PatternFingerprint(mol)
        combined = fp if combined is None else (combined | fp)
    return combined if combined is not None else DataStructs.ExplicitBitVect(2048)


def _pattern_fp_from_smiles_side(side: str) -> Any | None:
    combined = None
    for part in side.split("."):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            continue
        fp = Chem.PatternFingerprint(mol)
        combined = fp if combined is None else (combined | fp)
    return combined


def _one_to_one_match(smarts_mols: tuple[Any, ...], target_mols: list[Any]) -> bool:
    if len(smarts_mols) != len(target_mols):
        return False
    n = len(smarts_mols)
    for perm in itertools.permutations(range(n)):
        if all(
            target_mols[perm[i]].HasSubstructMatch(smarts_mols[i]) for i in range(n)
        ):
            return True
    return False


def _product_overlaps_reactants(
    product: str, reactants: Optional[tuple[str, ...]]
) -> bool:
    return bool(reactants) and product in reactants


def _normalize_retro_template(template: str) -> str:
    left = template.split(">>", 1)[0]
    if len([part for part in left.split(".") if part.strip()]) > 1:
        return "(" + template.replace(">>", ")>>", 1)
    return template


def _mols_from_smiles_side(side: str) -> list[Any] | None:
    mols = []
    for part in side.split("."):
        part = part.strip()
        if not part:
            continue
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            return None
        mols.append(mol)
    return mols or None


def _smarts_from_record(record: Any) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for field in ("key", "smarts", "reaction_smarts", "template"):
            value = record.get(field)
            if isinstance(value, str):
                return value
    return ""


def _iter_template_smarts(template_path: str):
    """Yield (scan_index, SMARTS) from WanJ jsonl or DESP idx2template json."""
    if template_path.endswith(".json") and not template_path.endswith(".jsonl"):
        with open(template_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            records = payload.values()
        elif isinstance(payload, list):
            records = payload
        else:
            raise ValueError(f"unsupported template json: {template_path}")
        for index, record in enumerate(records, start=1):
            yield index, _smarts_from_record(record)
        return
    with open(template_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            yield line_no, _smarts_from_record(json.loads(line))


@functools.lru_cache(maxsize=4)
def _load_template_keys(template_path: str) -> frozenset[str]:
    templates: set[str] = set()
    for _, key in _iter_template_smarts(template_path):
        if isinstance(key, str) and (normalized := _canonicalize_template(key)):
            templates.add(normalized)
    if not templates:
        raise ValueError(f"template library is empty: {template_path}")
    return frozenset(templates)


@functools.lru_cache(maxsize=4)
def _load_apply_library(template_path: str) -> ApplyLibrary:
    smarts_list: list[str] = []
    product_fps: list[Any] = []
    reactant_fps: list[Any] = []
    skipped = 0
    for line_no, key in _iter_template_smarts(template_path):
        if not isinstance(key, str) or ">>" not in key:
            skipped += 1
            continue
        product_smarts, reactant_smarts = key.split(">>", 1)
        smarts_list.append(key)
        product_fps.append(_pattern_fp_from_smarts_side(product_smarts))
        reactant_fps.append(_pattern_fp_from_smarts_side(reactant_smarts))
        if line_no % 50000 == 0:
            gc.collect()
            logger.info(
                "rdchiral apply index building | scanned=%d kept=%d",
                line_no,
                len(smarts_list),
            )
    if not smarts_list:
        raise ValueError(f"apply-match library is empty: {template_path}")
    logger.info(
        "rdchiral apply index ready | templates=%d skipped=%d",
        len(smarts_list),
        skipped,
    )
    return ApplyLibrary(
        smarts=tuple(smarts_list),
        product_fps=tuple(product_fps),
        reactant_fps=tuple(reactant_fps),
    )


@functools.lru_cache(maxsize=1)
def _load_rxnmapper():
    from rxnmapper import RXNMapper

    return RXNMapper()


@functools.lru_cache(maxsize=1)
def _load_rdchiral_extractor():
    from rdchiral import template_extractor as rdchiral_template_extractor

    return rdchiral_template_extractor.extract_from_reaction


class RdchiralTemplateFilter:
    """Validate a predicted reaction against the WanJ rdchiral jsonl library."""

    gate_name = "template_filter"

    def __init__(
        self, config: dict[str, Any], *, starting_mols: Container[str] | None = None
    ) -> None:
        backend = str(config.get("backend") or "rdchiral").strip().lower()
        if backend != "rdchiral":
            raise ValueError(
                f"template_filter.backend must be 'rdchiral', got {backend!r}"
            )
        self.backend = backend
        self.match_mode = _parse_mode(config.get("match_mode"))
        raw_path = config.get("template_path") or _WANJ_RDCHIRAL_JSONL
        self.template_path = _resolve_path(str(raw_path))
        self.mapper_timeout = float(config.get("mapper_timeout", 60))
        self.apply_max_templates = int(config.get("apply_max_templates", 256))
        self.blurry = _as_bool(config.get("blurry"), default=True)
        self.blurry_top_n = int(config.get("blurry_top_n", 1000))
        self.ranking = str(config.get("ranking", "aot")).strip().lower()
        if self.ranking not in {"aot", "none"}:
            raise ValueError("template_filter.ranking must be 'aot' or 'none'")
        self.unsolved_penalty = float(config.get("unsolved_penalty", 0.1))
        if not 0 <= self.unsolved_penalty < float("inf"):
            raise ValueError("template_filter.unsolved_penalty must be finite and nonnegative")
        if self.apply_max_templates < 1 or self.blurry_top_n < 1:
            raise ValueError("template limits must be positive")
        self._cand_workers_cfg = config.get("cand_workers")
        self._tmpl_workers_cfg = config.get("tmpl_workers")
        if not self.template_path:
            raise FileNotFoundError(
                "rdchiral template library not found; set template_path in the "
                "config or export WANJ_RDCHIRAL_TEMPLATE_JSONL (tried: "
                f"{raw_path or '<unset>'})"
            )
        self.ranker = None
        if self.match_mode == "apply" and self.ranking == "aot":
            self.ranker = AOTReactantRanker(
                starting_mols if starting_mols is not None else frozenset(),
                scscore_weights=config.get("scscore_weights"),
            )
        self._run_retro_outputs = functools.lru_cache(maxsize=8192)(self._run_retro_outputs)
        if self.match_mode == "apply":
            n_templates = len(_load_apply_library(self.template_path).smarts)
        else:
            n_templates = len(_load_template_keys(self.template_path))
        logger.info(
            "rdchiral template filter ready | mode=%s | blurry=%s top_n=%d | ranking=%s gamma=%.3f stock=%d | path=%s | templates=%d | timeout=%.0fs | cand_workers=%d tmpl_workers=%d",
            self.match_mode,
            self.blurry,
            self.blurry_top_n,
            self.ranking,
            self.unsolved_penalty,
            len(starting_mols) if starting_mols is not None else 0,
            self.template_path,
            n_templates,
            self.mapper_timeout,
            _worker_count(self._cand_workers_cfg, "AOT_GATE_CAND_WORKERS", 10, 10**9),
            _worker_count(self._tmpl_workers_cfg, "AOT_GATE_TMPL_WORKERS", 1, 10**9),
        )

    def _map_reaction(self, reaction_smiles: str) -> tuple[str, str]:
        mapper = _load_rxnmapper()
        mapped_reaction = mapper.get_attention_guided_atom_maps(
            [reaction_smiles], canonicalize_rxns=False
        )[0]["mapped_rxn"]
        reactants, products = mapped_reaction.split(">>", 1)
        return reactants, products

    def _extract_template(self, reaction_smiles: str) -> str:
        reactants, products = self._map_reaction(reaction_smiles)
        extract_from_reaction = _load_rdchiral_extractor()
        template_info = extract_from_reaction(
            {"_id": 0, "reactants": reactants, "products": products}
        )
        return _canonicalize_template(str((template_info or {}).get("reaction_smarts") or ""))

    def _run_retro_outputs(self, product: str, template_smarts: str) -> list[list[str]]:
        from rdchiral.initialization import rdchiralReactants, rdchiralReaction
        from rdchiral.main import rdchiralRun

        compiled = rdchiralReaction(_normalize_retro_template(template_smarts))
        outputs = rdchiralRun(compiled, rdchiralReactants(product))
        return [part.split(".") for part in outputs if part]

    def _run_retro(self, product: str, template_smarts: str) -> bool:
        return bool(self._run_retro_outputs(product, template_smarts))

    def _blurry_ranked_indices(
        self,
        reactants: str,
        library: ApplyLibrary,
        ranked: list[tuple[float, int]],
    ) -> list[int]:
        pool_indices = [index for _, index in ranked]
        if not pool_indices:
            return []
        llm_fp = _pattern_fp_from_smiles_side(reactants)
        if llm_fp is None or not library.reactant_fps:
            return pool_indices[: self.blurry_top_n]
        product_sims = {index: score for score, index in ranked}
        pool_fps = [library.reactant_fps[i] for i in pool_indices]
        reactant_scores = DataStructs.BulkTverskySimilarity(llm_fp, pool_fps, 0, 1)
        ordered_pos = sorted(
            range(len(pool_indices)),
            key=lambda pos: (
                float(reactant_scores[pos]),
                product_sims[pool_indices[pos]],
            ),
            reverse=True,
        )[: self.blurry_top_n]
        return [pool_indices[pos] for pos in ordered_pos]

    def _blurry_search(
        self,
        product: str,
        reactants: str,
        library: ApplyLibrary,
        ranked: list[tuple[float, int]],
    ) -> FilterResult:
        templates = [
            library.smarts[index]
            for index in self._blurry_ranked_indices(reactants, library, ranked)
        ]
        result = self._select_outputs(product, templates, "blurry")
        return result or FilterResult("FAIL", self.gate_name, "blurry search found no template")

    def _map_retro_outputs(
        self, product: str, templates: list[str], detail: str
    ) -> list[list[list[str]] | None]:
        def one(template_smarts: str) -> list[list[str]] | None:
            try:
                return self._run_retro_outputs(product, template_smarts)
            except Exception as exc:
                logger.debug("rdchiralRun failed for %s: %r", detail, exc)
                return None

        if not templates:
            return []
        workers = _worker_count(
            self._tmpl_workers_cfg, "AOT_GATE_TMPL_WORKERS", 1, len(templates)
        )
        if workers == 1:
            return [one(template_smarts) for template_smarts in templates]
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="aot-tmpl"
        ) as pool:
            futs = [pool.submit(one, template_smarts) for template_smarts in templates]
            return [fut.result() for fut in futs]

    def _select_outputs(
        self, product: str, templates: list[str], detail: str
    ) -> FilterResult | None:
        mapped = self._map_retro_outputs(product, templates, detail)
        best = None
        best_value = float("-inf")
        seen = set()
        for outputs in mapped:
            if not outputs:
                continue
            for output in outputs if self.ranker is not None else outputs[:1]:
                rewritten = _canonical_reactant_tuple(output)
                if rewritten is None:
                    continue
                key = tuple(sorted(rewritten))
                if key in seen:
                    continue
                seen.add(key)
                if self.ranker is None:
                    return FilterResult("PASS", self.gate_name, detail, reactants=rewritten)
                reward = self.ranker.score(rewritten)
                value = self.ranker.score(rewritten, self.unsolved_penalty) if detail == "blurry" else reward
                if value > best_value:
                    best_value = value
                    best = FilterResult(
                        "PASS", self.gate_name, detail, reactants=rewritten, rank_score=reward
                    )
                # All-stock is the maximum possible AOT reward; ties keep file order.
                if value == 0.0:
                    return best
        return best

    def _apply_match(self, reactants: str, product: str) -> FilterResult:
        product_mols = _mols_from_smiles_side(product)
        reactant_mols = _mols_from_smiles_side(reactants)
        if product_mols is None or reactant_mols is None:
            return FilterResult("UNKNOWN", self.gate_name, "invalid canonical input")
        if len(product_mols) != 1:
            return FilterResult("FAIL", self.gate_name, "apply-match expects one product")

        library = _load_apply_library(self.template_path)
        product_fp = Chem.PatternFingerprint(product_mols[0])
        scores = DataStructs.BulkTverskySimilarity(product_fp, list(library.product_fps), 0, 1)
        ranked = [
            (float(score), index)
            for index, score in enumerate(scores)
            if float(score) >= 1.0 - 1e-9
        ]
        ranked.sort(key=lambda item: item[0], reverse=True)

        matched_smarts: list[str] = []
        for _, index in ranked:
            product_smarts, reactant_smarts = library.smarts[index].split(">>", 1)
            template_product_mols = _split_smarts_mols(product_smarts)
            template_reactant_mols = _split_smarts_mols(reactant_smarts)
            if template_product_mols is None or template_reactant_mols is None:
                continue
            if not _one_to_one_match(template_product_mols, product_mols):
                continue
            if not _one_to_one_match(template_reactant_mols, reactant_mols):
                continue
            matched_smarts.append(library.smarts[index])
            if len(matched_smarts) >= self.apply_max_templates:
                break

        if matched_smarts:
            result = self._select_outputs(product, matched_smarts, "apply-match")
            if result is not None:
                if _product_overlaps_reactants(product, result.reactants):
                    return FilterResult(
                        "FAIL", self.gate_name, "product overlaps reactants"
                    )
                return result
        elif not self.blurry:
            return FilterResult("FAIL", self.gate_name, "no template matches reaction")

        if self.blurry:
            result = self._blurry_search(product, reactants, library, ranked)
            if result.status == "PASS" and _product_overlaps_reactants(
                product, result.reactants
            ):
                return FilterResult(
                    "FAIL", self.gate_name, "product overlaps reactants"
                )
            return result
        return FilterResult("FAIL", self.gate_name, "matched template does not apply")

    def _exact_match(self, reactants: str, product: str) -> FilterResult:
        templates = _load_template_keys(self.template_path)
        with _TEMPLATE_LOCK:
            future = _TEMPLATE_EXECUTOR.submit(
                self._extract_template, f"{reactants}>>{product}"
            )
            template = _canonicalize_template(future.result(timeout=self.mapper_timeout))
        if not template:
            return FilterResult("FAIL", self.gate_name, "template is absent from library")
        if template in templates:
            return FilterResult("PASS", self.gate_name)
        return FilterResult("FAIL", self.gate_name, "template is absent from library")

    def validate(self, *, reactants: list[str], product: str) -> FilterResult:
        canonical_reactants = canonicalize_reactants(".".join(reactants))
        canonical_product = canonicalize_smiles_clear_map(product)
        if not canonical_reactants or not canonical_product:
            return FilterResult("UNKNOWN", self.gate_name, "invalid canonical input")

        try:
            if self.match_mode == "apply":
                return self._apply_match(canonical_reactants, canonical_product)
            return self._exact_match(canonical_reactants, canonical_product)
        except FuturesTimeoutError:
            logger.warning("%s %s timed out; dropping candidate", self.backend, self.match_mode)
            return FilterResult("FAIL", self.gate_name, "mapper timeout")
        except Exception as exc:
            logger.warning(
                "%s %s failed; dropping candidate: %r",
                self.backend,
                self.match_mode,
                exc,
            )
            return FilterResult("FAIL", self.gate_name, type(exc).__name__)

    def _validate_candidates(
        self, product: str, candidates: list[list[str]]
    ) -> list[FilterResult]:
        if not candidates:
            return []
        if len(candidates) == 1:
            return [self.validate(reactants=list(candidates[0]), product=product)]
        workers = _worker_count(
            self._cand_workers_cfg, "AOT_GATE_CAND_WORKERS", 10, len(candidates)
        )
        if workers == 1:
            return [
                self.validate(reactants=list(reactants), product=product)
                for reactants in candidates
            ]
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="aot-cand"
        ) as pool:
            futs = [
                pool.submit(
                    self.validate, reactants=list(reactants), product=product
                )
                for reactants in candidates
            ]
            return [fut.result() for fut in futs]

    def filter_candidates(
        self, *, product: str, candidates: list[list[str]]
    ) -> list[FilterResult]:
        """AOT* gate for a batch of single-step proposals.

        Each candidate goes through apply-match then blurry. Surviving splits
        are deduplicated and sorted by AOT reward when ranking is on.

        Candidates in one request are gated in parallel, then gathered in the
        original order before dedup / ranking. Same per-candidate rule as the
        serial path. ``AOT_GATE_CAND_WORKERS`` caps the pool (default 10).
        """
        kept: list[FilterResult] = []
        seen: set[tuple[str, ...]] = set()
        results = self._validate_candidates(product, candidates)
        for result in results:
            if result.status != "PASS" or not result.reactants:
                continue
            key = tuple(sorted(result.reactants))
            if key in seen:
                continue
            seen.add(key)
            kept.append(result)
        if kept and all(item.rank_score is not None for item in kept):
            kept.sort(key=lambda item: item.rank_score or 0.0, reverse=True)
        return kept


def build_rdchiral_template_filter(
    config: Optional[dict[str, Any]],
    *, starting_mols: Container[str] | None = None,
) -> Optional[RdchiralTemplateFilter]:
    if not config or not config.get("enabled"):
        return None
    return RdchiralTemplateFilter(config, starting_mols=starting_mols)
