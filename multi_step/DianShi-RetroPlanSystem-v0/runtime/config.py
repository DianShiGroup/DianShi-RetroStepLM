import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

_ENV_REFERENCE = re.compile(
    r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}$"
    r"|^\$(?P<plain>[A-Za-z_][A-Za-z0-9_]*)$"
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE file without overriding existing values."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_env_value(value: Any) -> Any:
    """Resolve ``$VAR``, ``${VAR}`` and ``${VAR:-default}`` references to the environment."""
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    match = _ENV_REFERENCE.match(value)
    if match is None:
        return os.environ.get(value[1:], "")
    if match.group("plain"):
        return os.environ.get(match.group("plain"), "")
    default = match.group("default") if match.group("default") is not None else ""
    return os.environ.get(match.group("name"), default)


def _resolve_env_list(value: Any) -> Any:
    if isinstance(value, list):
        return [_resolve_env_value(item) for item in value]
    return _resolve_env_value(value)


def load_config(config_path: str) -> Dict[str, Any]:
    _load_env_file(_REPO_ROOT / ".env")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["starting_molecules"] = _resolve_env_value(config.get("starting_molecules"))

    for section in ["planner", "dianshi", "vllm"]:
        if section in config:
            for key in ["api_key", "api_base", "api_url", "base_url"]:
                config[section][key] = _resolve_env_value(config[section].get(key))

    single_step = config.get("single_step")
    if isinstance(single_step, dict):
        single_step["api_key"] = _resolve_env_value(single_step.get("api_key"))
        single_step_retro = single_step.get("retro")
        if isinstance(single_step_retro, dict):
            single_step_retro["api_url"] = _resolve_env_list(single_step_retro.get("api_url"))
            single_step_retro["model"] = _resolve_env_value(single_step_retro.get("model"))
        template_filter = single_step.get("template_filter")
        if isinstance(template_filter, dict):
            template_filter["template_path"] = _resolve_env_value(
                template_filter.get("template_path")
            )

    llm_cfg = config.get("llm_config")
    if llm_cfg:
        providers = llm_cfg.get("providers")
        if isinstance(providers, dict):
            for provider_cfg in providers.values():
                if isinstance(provider_cfg, dict):
                    for key in ["api_key", "base_url"]:
                        provider_cfg[key] = _resolve_env_value(provider_cfg.get(key))
                    for model_cfg in (provider_cfg.get("llms") or {}).values():
                        if isinstance(model_cfg, dict):
                            for key in ["api_key", "base_url"]:
                                model_cfg[key] = _resolve_env_value(model_cfg.get(key))
                            model_cfg["model_name"] = _resolve_env_value(
                                model_cfg.get("model_name")
                            )

    return config
