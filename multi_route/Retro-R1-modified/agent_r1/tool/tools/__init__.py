"""
Specific tool implementations
"""

from agent_r1.tool.tools.search_tool import SearchTool
from agent_r1.tool.tools.calculator_tool import CalculatorTool
from agent_r1.tool.tools.wiki_search_tool import WikiSearchTool
from agent_r1.tool.tools.single_step_retro_tool import SingleStepRetroTool
from agent_r1.tool.tools.single_step_retro_vllm_tool import SingleStepRetroVLLMTool, create_vllm_tool
from agent_r1.tool.tools.select_reaction_tool import SelectReactionTool
from agent_r1.tool.tools.back_state_tool import BackStateTool



__all__ = [
    'SearchTool',
    'CalculatorTool',
    'WikiSearchTool',
    'SingleStepRetroTool',
    'SingleStepRetroVLLMTool',
    'create_vllm_tool',
    'SelectReactionTool',
    'BackStateTool',
]

def _default_tools(env, **kwargs):
    """
    Create default tools based on environment type.

    Args:
        env: Environment name (e.g., 'retro', 'retro_noback', 'retro_vllm')
        **kwargs: Additional arguments for vLLM tool:
            - vllm_api_url: URL for vLLM API (default: "http://localhost:8000/v1/chat/completions")
            - model_name: Model name for API (default: "default")
            - vllm_topk: Number of reactions to return (default: 10)
            - vllm_temperature: Temperature for sampling (default: 0.0)
            - vllm_max_tokens: Max tokens to generate (default: 1024)
            - vllm_beam_width: Beam width for beam search (default: 5)
    """
    if env == 'search':
        return [SearchTool()]
    elif env == 'calculator':
        return [CalculatorTool()]
    elif env == 'wikisearch':
        return [WikiSearchTool()]
    elif env == 'retro':
        return [SingleStepRetroTool(), SelectReactionTool(), BackStateTool()]
    elif env == 'retro_noback':
        return [SingleStepRetroTool(), SelectReactionTool()]
    elif env == 'retro_noback_V4':
        return [SingleStepRetroTool(mlp_model_dump='./one_step_model/retro_star_V4.ckpt'), SelectReactionTool()]
    elif env == 'retro_noback_V3':
        return [SingleStepRetroTool(mlp_model_dump='./one_step_model/retro_star_zero_ours.ckpt'), SelectReactionTool()]
    elif env == 'retro_noback_V2':
        return [SingleStepRetroTool(mlp_model_dump='./one_step_model/retro_star_value_ours.ckpt'), SelectReactionTool()]
    # vLLM-based single step retro options
    elif env.startswith('retro_vllm'):
        # Extract variant from env name (e.g., 'retro_vllm' -> default, 'retro_vllm_cot' -> COT variant)
        variant = env.replace('retro_vllm_', '') if 'retro_vllm_' in env else None
        vllm_tool = create_vllm_tool(
            vllm_api_url=kwargs.get('vllm_api_url', "http://localhost:8000/v1/chat/completions"),
            model_name=kwargs.get('model_name', "default"),
            topk=kwargs.get('vllm_topk', 10),
            temperature=kwargs.get('vllm_temperature', 0.0),
            max_tokens=kwargs.get('vllm_max_tokens', 1024),
            beam_width=kwargs.get('vllm_beam_width', 5),
        )
        return [vllm_tool, SelectReactionTool()]
    else:
        raise NotImplementedError(f"Environment '{env}' not implemented. Available: search, calculator, wikisearch, retro, retro_noback, retro_noback_V2, retro_noback_V3, retro_noback_V4, retro_vllm")
