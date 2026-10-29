"""Prompt-section parsing and message building for the single-step models.

`load_prompt_sections` parses ``---SYSTEM---`` / ``---USER---`` prompt files;
`build_rag_messages` renders the single-step prompt (with an empty retrieval
result it is the plain no-RAG prompt) and `build_forward_messages` renders the
forward round-trip gate prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_format_keys(text: str) -> set[str]:
    return set(re.findall(r"\{(\w+)", text)) if text else set()


def load_prompt_sections(txt_path: str) -> dict[str, str]:
    """Parse a prompt file containing ---SYSTEM--- / ---USER--- sections."""
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    sections = {"system": "", "user": ""}
    current = None
    buf: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "---SYSTEM---":
            if current:
                sections[current] = "\n".join(buf).strip()
            current, buf = "system", []
        elif stripped == "---USER---":
            if current:
                sections[current] = "\n".join(buf).strip()
            current, buf = "user", []
        elif stripped == "---ASSISTANT---":
            if current:
                sections[current] = "\n".join(buf).strip()
            current = None
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def build_rag_examples_str(rag_results: list[dict[str, Any]], max_examples: int) -> str:
    """Format retrieved examples for prompt insertion."""
    if not isinstance(rag_results, list):
        return ""

    examples = []
    for i, ex in enumerate(rag_results):
        if i >= max_examples:
            break
        if not isinstance(ex, dict):
            continue

        reactants = ex.get("reactants", "")
        products = ex.get("products", "")
        leaving_groups = ex.get("leaving_groups", [])
        synthons_smiles = ex.get("synthons_smiles", [])

        if isinstance(reactants, list):
            reactants = ".".join(str(x) for x in reactants)
        reactants = str(reactants)

        if isinstance(products, list):
            products = ".".join(str(x) for x in products)
        products = str(products)

        if not isinstance(leaving_groups, list):
            leaving_groups = (
                [leaving_groups] if leaving_groups not in (None, "") else []
            )
        if not isinstance(synthons_smiles, list):
            synthons_smiles = (
                [synthons_smiles] if synthons_smiles not in (None, "") else []
            )

        examples.append(
            "\n".join(
                [
                    f"Example {i + 1}",
                    f"reactants: {reactants}",
                    f"products: {products}",
                    f"leaving_groups: {json.dumps(leaving_groups, ensure_ascii=False)}",
                    f"synthons_smiles: {json.dumps(synthons_smiles, ensure_ascii=False)}",
                ]
            )
        )
    return "\n\n".join(examples).strip()


def build_rag_messages(
    product_smiles: str,
    prompt_sections: dict[str, str],
    rag_results: list[dict[str, Any]],
    max_examples: int,
) -> list[dict[str, str]]:
    """Build chat messages for the single-step vLLM call.

    With an empty ``rag_results`` this is the plain no-RAG prompt.
    """
    rag_examples_str = build_rag_examples_str(rag_results, max_examples=max_examples)
    template_vars = {
        "product_smiles": product_smiles,
        "products": product_smiles,
        "rag_examples": rag_examples_str,
        "reactant_smiles": "",
        "reactants": "",
        "synthons_str": "[]",
        "synthons_smiles": "[]",
        "leaving_groups_str": "[]",
        "leaving_groups": "[]",
        "error_res": "",
        "product_bond_str": "[]",
        "reactant_fragments_str": "[]",
        "products_i": "",
        "reactants_i": "",
        "synthons_smiles_i": "",
        "leaving_groups_i": "",
        "similarity_i": "",
    }

    if prompt_sections is None:
        return []

    required_keys: set[str] = set()
    for section in prompt_sections.values():
        required_keys |= extract_format_keys(section)
    missing = required_keys - set(template_vars.keys())
    if missing:
        raise KeyError(
            f"Prompt placeholders were not found in template vars: {sorted(missing)}"
        )

    return [
        {"role": "system", "content": prompt_sections["system"]},
        {"role": "user", "content": prompt_sections["user"].format(**template_vars)},
    ]


def build_forward_messages(
    reactants: str,
    prompt_sections: dict[str, str],
    rag_results: list[dict[str, Any]],
    max_examples: int,
) -> list[dict[str, str]]:
    """Build the forward-model prompt from the configured template."""
    if prompt_sections is None:
        return []

    template_vars = {
        "reactants": reactants,
        "rag_examples": build_rag_examples_str(rag_results, max_examples=max_examples),
    }
    required_keys: set[str] = set()
    for section in prompt_sections.values():
        required_keys |= extract_format_keys(section)
    missing = required_keys - set(template_vars)
    if missing:
        raise KeyError(
            f"Forward prompt placeholders were not found in template vars: {sorted(missing)}"
        )

    return [
        {"role": "system", "content": prompt_sections["system"]},
        {"role": "user", "content": prompt_sections["user"].format(**template_vars)},
    ]
