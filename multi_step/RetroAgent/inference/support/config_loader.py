"""YAML config loading with environment-variable expansion.

Config files may reference environment variables as ``${VAR}`` or
``${VAR:-fallback}``. This lets a checkout point at its own checkpoints, RAG
library and model servers without editing tracked config files: keep the real
values in a local ``.env`` and ship the same file as ``.env.example``.

An unset ``${VAR}`` with no fallback is left untouched, so a typo shows up as a
literal string in the error message the caller already raises, rather than
silently becoming an empty path.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    """Substitute ``${VAR}`` / ``${VAR:-fallback}`` from the environment."""

    def _replace(match: "re.Match[str]") -> str:
        name, fallback = match.group(1), match.group(2)
        value = os.environ.get(name)
        if value:
            return value
        if fallback is not None:
            return fallback
        return match.group(0)

    return _ENV_REF.sub(_replace, text)


def load_yaml(path: "str | Path") -> Any:
    """Read a YAML file after expanding environment references."""
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(expand_env(text))