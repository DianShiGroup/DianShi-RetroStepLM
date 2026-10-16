"""DianShi direct single-step adapter: an OpenAI-compatible HTTP step model
with the rdchiral/AOT* template gate, no retrieval."""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


from data.building_blocks import BuildingBlockChecker

from inference.support.config_loader import load_yaml
from inference.support.prompt_sections import build_direct_messages, load_prompt_sections
from inference.support.smiles_canonical import (
    canonicalize_reactants,
    canonicalize_smiles_clear_map,
)
from inference.gate.rdchiral_template_filter import build_rdchiral_template_filter
from inference.support.smiles_atomwise import restore_answer_spaces, transform_messages


logger = logging.getLogger(__name__)
ANSWER_RE = re.compile(r"<answer\s*>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
DEFAULT_STARTING_MOLECULES = str(
    Path(__file__).resolve().parents[2] / "dataset" / "emolecules_standardized.sqlite"
)


def _load_starting_molecules(path: str):
    if not path or not os.path.isfile(path):
        logger.warning("starting molecules file not found: %s", path)
        return frozenset()
    lowered = path.lower()
    if lowered.endswith(".sqlite") or lowered.endswith(".db"):
        return BuildingBlockChecker(path)
    mols: set[str] = set()
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "mol" not in reader.fieldnames:
            raise ValueError("starting_molecules csv must have a mol column: %s" % path)
        for row in reader:
            smiles = (row.get("mol") or "").strip()
            if smiles:
                mols.add(smiles)
    return frozenset(mols)


def _resolve_path(project_root: Path, value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else project_root / path)


def _load_config(config_path: str) -> Tuple[Dict[str, Any], Path]:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Single-step config does not exist: %s" % path)
    config = load_yaml(path)
    if not isinstance(config, dict):
        raise ValueError("Single-step config must contain a YAML mapping")
    return config, path


class DianShiDirectStep:
    """Return vLLM predictions in the legacy MLP result format."""

    def __init__(self, config_path: str) -> None:
        config, path = _load_config(config_path)
        project_root = (
            path.parent.parent if path.parent.name == 'configs' else path.parent
        )
        retro = config.get("retro") or {}
        urls = retro.get("api_url") or retro.get("api_urls")
        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not all(isinstance(url, str) and url for url in urls):
            raise ValueError("retro.api_url must contain at least one endpoint")

        self.api_urls = urls
        self.model_name = str(retro.get("model") or "")
        self.num_res = int(retro.get("num_res") or 10)
        if not self.model_name or self.num_res < 1:
            raise ValueError("retro.model and a positive retro.num_res are required")

        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 2048))
        self.timeout = int(config.get("timeout", 300))
        self.atomwise = bool(config.get("atomwise", False))
        self.api_key = str(config.get("api_key") or os.environ.get("DIANSHI_DIRECT_API_KEY", ""))

        prompt_path = config.get("prompt_path_no_rag")
        if not prompt_path:
            raise ValueError("prompt_path_no_rag is required")

        self.prompt_sections = load_prompt_sections(
            _resolve_path(project_root, prompt_path)
        )
        template_filter = config.get("template_filter") or {}
        starting_path = str(
            template_filter.get("starting_molecules") or DEFAULT_STARTING_MOLECULES
        )
        self.starting_mols = _load_starting_molecules(starting_path)
        self.template_filter = build_rdchiral_template_filter(
            template_filter, starting_mols=self.starting_mols
        )
        if self.template_filter is not None:
            logger.info(
                "DianShi-Direct single-step rdchiral template_filter=on | stock=%d",
                len(self.starting_mols),
            )

    def _messages(self, smiles: str) -> List[Dict[str, str]]:
        messages = build_direct_messages(
            product_smiles=smiles,
            prompt_sections=self.prompt_sections,
        )
        if self.atomwise:
            messages, _ = transform_messages(messages)
        return messages

    def _request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_beam_search": True,
            "best_of": self.num_res,
            "n": self.num_res,
        }
        request = Request(
            random.choice(self.api_urls),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer %s" % self.api_key,
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("vLLM response is not a JSON object")
        return result

    def _predict(
        self, smiles: str, canonical_smiles: str
    ) -> Optional[List[Dict[str, Any]]]:
        messages = self._messages(canonical_smiles)
        for attempt in range(3):
            try:
                raw = self._request(messages)
                predictions = []
                for rank, choice in enumerate(raw.get("choices", []), start=1):
                    if not isinstance(choice, dict):
                        continue
                    content = choice.get("message", {}).get("content", "") or ""
                    if self.atomwise:
                        content = restore_answer_spaces(content).text
                    match = ANSWER_RE.search(content)
                    answer = match.group(1).strip() if match else ""
                    if answer:
                        predictions.append(
                            {
                                "rank": rank,
                                "output_smiles": answer,
                                "parsed": [
                                    part.strip()
                                    for part in answer.split(".")
                                    if part.strip()
                                ],
                            }
                        )
                    else:
                        predictions.append(
                            {
                                "rank": rank,
                                "output_smiles": "",
                                "parsed": [],
                            }
                        )
                if any(item["parsed"] for item in predictions):
                    return predictions
            except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError):
                logger.warning(
                    "DianShi-Direct single-step call failed (%d/3) for %s",
                    attempt + 1,
                    smiles[:80],
                    exc_info=True,
                )
            if attempt < 2:
                time.sleep(2 ** attempt)
        return None

    def _to_legacy_result(
        self,
        predictions: List[Dict[str, Any]],
        topk: int,
        product: str,
    ) -> Optional[Dict[str, Any]]:
        parsed_candidates: List[List[str]] = []
        seen = set()
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            parsed = prediction.get("parsed")
            if isinstance(parsed, list) and parsed:
                answer = ".".join(str(part) for part in parsed if str(part).strip())
            else:
                answer = prediction.get("output_smiles", "")
            canonical = canonicalize_reactants(answer)
            if not canonical or canonical in seen:
                continue
            seen.add(canonical)
            parsed_candidates.append(canonical.split("."))
        if self.template_filter is not None:
            filtered = self.template_filter.filter_candidates(
                product=product, candidates=parsed_candidates
            )
            candidates = []
            kept = set()
            for result in filtered:
                if not result.reactants:
                    continue
                canonical = canonicalize_reactants(".".join(result.reactants))
                if not canonical or canonical in kept:
                    continue
                kept.add(canonical)
                candidates.append(canonical)
        else:
            candidates = [".".join(parts) for parts in parsed_candidates]
        reactants = candidates[:topk]
        if not reactants:
            return None

        templates = [None] * len(reactants)
        return {
            "reactants": reactants,
            "template": templates,
            "templates": templates,
            "templates_idx": [-1] * len(reactants),
        }

    def run(self, smiles: Any, topk: int = 10) -> Optional[Dict[str, Any]]:
        if topk < 1 or topk > self.num_res:
            raise ValueError("topk must be between 1 and %d" % self.num_res)
        smiles = getattr(smiles, "mol", smiles)
        if not isinstance(smiles, str):
            smiles = str(smiles)
        canonical_smiles = canonicalize_smiles_clear_map(smiles)
        if not canonical_smiles:
            logger.warning("Invalid single-step input SMILES: %s", smiles)
            return None

        predictions = self._predict(smiles, canonical_smiles)
        if predictions is None:
            return None
        return self._to_legacy_result(predictions, topk, canonical_smiles)


def load(config_path: str) -> DianShiDirectStep:
    return DianShiDirectStep(config_path)
