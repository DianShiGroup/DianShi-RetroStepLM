"""Prompt-file parsing and message assembly for the direct single-step call.

``load_prompt_sections`` splits a prompt file on ``---SYSTEM---`` /
``---USER---`` markers; ``build_direct_messages`` fills the placeholders those
sections declare. Neither knows anything about where the answers come from.

``build_direct_messages`` keeps the full placeholder set the step prompt has
ever used, with ``rag_examples`` pinned to the empty string. The shipped
``single_step_prompt_no_rag.txt`` only uses ``{products}``, ``{reactants}``,
``{synthons_smiles}`` and ``{leaving_groups}``, but leaving the rest in place
means pointing the config at a different prompt file renders blanks instead of
raising ``KeyError``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

_FORMAT_KEY_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def extract_format_keys(text: str) -> set[str]:
    """Return every ``{placeholder}`` name used in a prompt section."""
    return set(_FORMAT_KEY_RE.findall(str(text or "")))


def load_prompt_sections(txt_path: str) -> Dict[str, str]:
    """Parse a prompt file containing ---SYSTEM--- / ---USER--- sections."""
    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    sections = {"system": "", "user": ""}
    current = None
    buf: List[str] = []
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


def build_direct_messages(
    product_smiles: str,
    prompt_sections: Dict[str, str],
) -> List[Dict[str, str]]:
    """Build the chat messages for one direct single-step call."""
    if prompt_sections is None:
        return []

    template_vars: Dict[str, Any] = {
        "product_smiles": product_smiles,
        "products": product_smiles,
        "rag_examples": "",
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
