"""Build custom inference tools while reusing the original Retro-R1 tools."""

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

from inference.step.mlp import SingleStepRetroMLPTool
from inference.step.dianshi_direct import DianShiDirectStepRetroTool


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_original_class(relative_path: str, class_name: str):
    """Load one original tool without importing the whole tools package."""
    path = REPO_ROOT / relative_path
    module_name = f"retro_r1_inference_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def get_tools(
    backend: str = "dianshi_direct",
    step_api_url: str = "http://localhost:8000/v1/chat/completions",
    step_api_urls: Optional[List[str]] = None,
    model_name: str = "default",
    topk: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    beam_width: int = 10,
    starting_molecules: str = "./dataset/origin_dict.csv",
    api_key: Optional[str] = None,
    timeout: int = 120,
    mlp_model_dump: Optional[str] = None,
    atomwise: bool = False,
    prompt_path_no_rag: Optional[str] = None,
    template_filter: Optional[Dict[str, Any]] = None,
):
    """Return one selected single-step tool and the original selector tool."""
    SelectReactionTool = _load_original_class(
        "agent_r1/tool/tools/select_reaction_tool.py", "SelectReactionTool"
    )

    if backend == "mlp":
        if not mlp_model_dump:
            raise ValueError("mlp_model_dump is required for the MLP backend")
        single_step = SingleStepRetroMLPTool(mlp_model_dump=mlp_model_dump)
    elif backend == "dianshi_direct":
        single_step = DianShiDirectStepRetroTool(
            step_api_url=step_api_url,
            step_api_urls=step_api_urls,
            model_name=model_name,
            topk=topk,
            temperature=temperature,
            max_tokens=max_tokens,
            beam_width=beam_width,
            starting_molecules=starting_molecules,
            api_key=api_key,
            timeout=timeout,
            atomwise=atomwise,
            prompt_path_no_rag=prompt_path_no_rag,
            template_filter=template_filter,
        )
    else:
        raise ValueError(f"Unknown single-step backend: {backend}")

    return [single_step, SelectReactionTool()]
