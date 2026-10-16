from typing import Any, Dict


def _get_providers(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    llm_config = config.get("llm_config")
    if not llm_config:
        return {}
    providers = llm_config.get("providers")
    if not isinstance(providers, dict):
        return {}
    return providers


def _get_llms(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for provider_name, provider_cfg in _get_providers(config).items():
        llms = provider_cfg.get("llms")
        if not isinstance(llms, dict):
            continue
        for model_key, model_cfg in llms.items():
            if isinstance(model_cfg, dict):
                result[f"{provider_name}/{model_key}"] = model_cfg
    return result


def resolve_llm_config(config: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """Merge provider defaults with model-level overrides."""
    if "/" not in model_id:
        raise KeyError(model_id)
    provider_name = model_id.split("/", 1)[0]
    providers = _get_providers(config)
    if provider_name not in providers:
        raise KeyError(model_id)

    model_configs = _get_llms(config)
    if model_id not in model_configs:
        raise KeyError(model_id)

    provider_cfg = providers[provider_name]
    model_cfg = model_configs[model_id]

    def resolved(key: str, default: Any = None) -> Any:
        value = model_cfg.get(key)
        return value if value is not None else provider_cfg.get(key, default)

    return {
        "base_url": resolved("base_url", ""),
        "api_key": resolved("api_key", ""),
        "model": str(model_cfg.get("model_name", model_id)),
        "temperature": float(resolved("temperature", 0.0)),
        # Temperature suppression is model-specific: a provider default must
        # not disable temperature for every model on that provider.
        "disable_temperature": model_cfg.get("disable_temperature") is True,
        "max_tokens": int(resolved("max_tokens", 4096)),
        "timeout": resolved("timeout"),
        "extra_body": (
            dict(model_cfg["extra_body"])
            if isinstance(model_cfg.get("extra_body"), dict)
            else {}
        ),
    }


def get_provider_config(config: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    resolved = resolve_llm_config(config, model_id)
    return {
        "base_url": resolved["base_url"],
        "api_key": resolved["api_key"],
        "temperature": resolved["temperature"],
        "disable_temperature": resolved["disable_temperature"],
        "max_tokens": resolved["max_tokens"],
        "timeout": resolved["timeout"],
    }


def get_planner_selectable_models(config: Dict[str, Any]) -> Dict[str, str]:
    return {str(k): str(v.get("model_name", k)) for k, v in _get_llms(config).items()}


def list_planner_model_ids_sorted(config: Dict[str, Any]) -> list[str]:
    return sorted(_get_llms(config).keys())


def default_planner_model_id(config: Dict[str, Any]) -> str:
    llms = _get_llms(config)
    if not llms:
        return ""
    key = (config.get("planner") or {}).get("default_model")
    if isinstance(key, str) and key in llms:
        return key
    if isinstance(key, str):
        folded = {model_id.lower(): model_id for model_id in llms}
        matched = folded.get(key.lower())
        if matched:
            return matched
    return ""


def api_model_for_id(config: Dict[str, Any], model_id: str) -> str:
    llms = _get_llms(config)
    if model_id not in llms:
        raise KeyError(model_id)
    return str(llms[model_id].get("model_name", model_id))


def default_planner_api_model(config: Dict[str, Any]) -> str:
    model_id = default_planner_model_id(config)
    if not model_id:
        raise ValueError("llm_config.providers is empty or not configured")
    return api_model_for_id(config, model_id)


def llm_extra_body(config: Dict[str, Any], model_id: str) -> dict:
    return dict(resolve_llm_config(config, model_id).get("extra_body") or {})
