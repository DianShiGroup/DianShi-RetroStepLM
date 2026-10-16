"""
Single step retro tool implementation using vLLM API
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from agent_r1.tool.tool_base import Tool
import random

logger = logging.getLogger(__name__)


# Default prompt template
DEFAULT_PROMPT_TEMPLATE = """
---SYSTEM---
You are an expert chemist specializing in retrosynthetic analysis. Given the SMILES representation of a product molecule, your task is to predict the SMILES of the reactants required to synthesize it.
Input:
The product SMILES.
Task requirements (step-by-step reasoning is required):
1.Infer the synthons based on the structure of the product.
2.Identify the corresponding leaving groups associated with these synthons.
3.Reconstruct the most plausible reactant molecules by combining the synthons with appropriate leaving groups.
You must explicitly show your reasoning step by step for each of the above stages.
Please output the reasoning process, and write the final reactant SMILES strictly in the following format:
<answer>REACTANT_SMILES</answer>
If multiple reactants are required, separate their SMILES using a period (.).
---USER---
The product SMILES is {products}.
---ASSISTANT---
<think>The product SMILES is {products}. By analyzing the product SMILES, the synthons involved in this reaction are identified as {synthons_smiles}.
Based on the product structure and the inferred synthons, the corresponding leaving groups for this reaction are deduced to be {leaving_groups}.
By combining the synthons with the identified leaving groups, the final reactant(s) are inferred as:{reactants}</think>
<answer>{reactants}</answer>
"""


class SingleStepRetroVLLMTool(Tool):
    """
    Tool for performing Single step retrosynthesis using vLLM API
    """

    def __init__(
        self,
        vllm_api_url: str = "http://localhost:8000/v1/chat/completions",
        prompt_template: str = None,
        model_name: str = "default",
        topk: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        starting_molecules: str = "./dataset/origin_dict.csv",
        beam_width: int = 5,
        vllm_api_key: str = None,
    ):
        """
        Initialize the vLLM-based single step retro tool

        Args:
            vllm_api_url: URL for vLLM API endpoint
            prompt_template: Path to prompt template file
            model_name: Name of the model for API calls
            topk: Number of reactions to return
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            timeout: Timeout for API calls in seconds
            starting_molecules: Path to file containing available starting molecules
            beam_width: Beam width for beam search (generates beam_width candidates, returns top-k)
            vllm_api_key: Optional bearer token; omit for deployments that do not require one
        """
        self.vllm_api_key = vllm_api_key
        name = "single_step_retro"
        description = "Perform single step retrosynthesis for a molecule using LLM and return several possible reactions to synthesize it. Note that the reactions may be incorrect and each reactant is marked as 'available' or 'unavailable'. The unavailable molecules have to be synthesized further."
        parameters = {
            "type": "object",
            "properties": {
                "molecule": {
                    "type": "string",
                    "description": "The ID for the molecule to be synthesized. For example, 0-0 ."
                }
            },
            "required": ["molecule"]
        }

        self.prompt_template = DEFAULT_PROMPT_TEMPLATE

        self.vllm_api_url = vllm_api_url
        self.model_name = model_name
        self.topk = topk
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.beam_width = beam_width

        # Load starting molecules (available building blocks)
        self.starting_mols = self._load_starting_molecules(starting_molecules)

        super().__init__(name, description, parameters)

    def _load_starting_molecules(self, path: str) -> set:
        """Load starting molecules from file"""
        if not os.path.exists(path):
            # Try alternative path from Retro-R1 root
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset", "origin_dict.csv")
            if os.path.exists(alt_path):
                path = alt_path
            else:
                return set()

        import pandas as pd
        if path.endswith('.csv'):
            df = pd.read_csv(path)
            return set(df['mol'].tolist())
        elif path.endswith('.pkl'):
            import pickle
            with open(path, 'rb') as f:
                mols = pickle.load(f)
                return set(mols)
        return set()

    def _call_vllm_api(self, product_smiles: str) -> List[tuple]:
        """
        Call vLLM API to get single step retrosynthesis results

        Args:
            product_smiles: The product SMILES string

        Returns:
            List of tuples (reactant_smiles, logprobs) for each beam
        """
        import requests

        # Format prompt with product SMILES
        prompt = self.prompt_template.replace("{products}", product_smiles)

        messages = [
            {"role": "system", "content": prompt.split("---USER---")[0].replace("---SYSTEM---", "").strip()},
            {"role": "user", "content": prompt.split("---USER---")[1].split("---ASSISTANT---")[0].strip()}
        ]

        # Build request payload with beam search parameters
        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add beam search parameters
        request_payload["use_beam_search"] = True
        request_payload["best_of"] = self.beam_width
        # When using beam search, return n results
        request_payload["n"] = self.beam_width

        headers = {"Content-Type": "application/json"}
        # Only send Authorization when a key was configured; plenty of local
        # vLLM deployments run without one.
        if self.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.vllm_api_key}"

        try:
            logger.info(f"Calling vLLM API {self.vllm_api_url}")
            response = requests.post(
                self.vllm_api_url,
                json=request_payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()

            # Extract generated reactant SMILES from responses
            reactant_predictions = []
            if "choices" in result:
                for choice in result["choices"]:
                    content = choice.get("message", {}).get("content", "")

                    # Try to extract logprobs if available
                    logprobs = None
                    if "logprobs" in choice and choice["logprobs"] is not None:
                        try:
                            # Get the average logprob of generated tokens
                            token_logprobs = choice["logprobs"].get("content", [])
                            if token_logprobs:
                                logprobs = sum(lp.get("logprob", 0) for lp in token_logprobs) / len(token_logprobs)
                        except:
                            pass

                    # Extract SMILES from <answer> tags
                    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                    if match:
                        answer_text = match.group(1).strip()
                        # Handle multiple reactants separated by '.'
                        reactant_predictions.append((answer_text, logprobs))

            return reactant_predictions

        except requests.exceptions.Timeout:
            logger.error(f"vLLM API timeout after {self.timeout}s for molecule: {product_smiles}")
            raise TimeoutError(f"vLLM API timeout after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM API error for molecule {product_smiles}: {str(e)}")
            raise ConnectionError(f"vLLM API error: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from vLLM for molecule {product_smiles}: {str(e)}")
            raise ValueError(f"Invalid JSON response from vLLM: {str(e)}")

    def _parse_reactants_to_result(self, reactant_smiles: str, score: float = None) -> List[List]:
        """
        Parse reactant SMILES string to the result format

        Args:
            reactant_smiles: Reactant SMILES string (can contain multiple separated by '.')
            score: Score for this reaction

        Returns:
            List of [[reactant_smiles, availability], ...] - flattened format compatible with MLP model
        """
        reactant_list = reactant_smiles.split('.') if reactant_smiles else []
        reactant_list = [r.strip() for r in reactant_list if r.strip()]

        reactant_list_exist = []
        for reactant in reactant_list:
            is_available = reactant in self.starting_mols
            reactant_list_exist.append([reactant, 'available' if is_available else 'unavailable'])

        # If score not provided, assign default score
        if score is None:
            score = 1.0 / (len(reactant_list_exist) + 1)

        # Return only the reactant list for compatibility with MLP model format
        # The environment expects to iterate directly over reactants, not a tuple
        return reactant_list_exist

    def execute(self, args: Dict) -> str:
        """
        Execute single step retrosynthesis using vLLM

        Args:
            args: Tool parameters, containing:
                - "molecule": molecule SMILES or ID

        Returns:
            Dictionary with results
        """
        mol = args.get("molecule", "").strip()
        topk = args.get("topk", self.topk)
        shuffle = args.get("shuffle", False)
        reaction_list = args.get("reaction_list", [])
        exist = args.get("exist", False)

        if exist:
            return {"results": reaction_list}

        if not mol:
            return {"error": "No molecule provided."}

        try:
            # Check if it's available molecule
            if mol in self.starting_mols:
                return {"results": []}

            # Call vLLM API
            reactant_predictions = self._call_vllm_api(mol)

            if not reactant_predictions:
                return {"results": []}

            # Process predictions
            reactants = []
            for pred in reactant_predictions:
                reactant_smiles, logprobs = pred
                # Use logprobs as score if available, otherwise use default
                if logprobs is not None:
                    score = logprobs
                else:
                    # Fallback: use position-based score
                    score = 1.0 / (len(reactants) + 1)
                result = self._parse_reactants_to_result(reactant_smiles, score)
                reactants.append(result)

            # Sort by score
            reactants_sorted = sorted(reactants, key=lambda x: x[-1], reverse=True)

            if shuffle:
                random.shuffle(reactants_sorted)

            return {"results": reactants_sorted[:topk]}

        except Exception as e:
            logger.error(f"Error in vLLM single step retro for molecule {mol}: {str(e)}", exc_info=True)
            raise ValueError(f"Error in vLLM single step retro for molecule {mol}: {str(e)}", exc_info=True)

    def calculate_reward(self, args: Dict, result: str) -> float:
        """
        Calculate reward for the tool execution

        Args:
            args: Tool parameters
            result: Tool execution result

        Returns:
            Reward value
        """
        try:
            result_obj = result
            if "results" in result_obj:
                return 0.0
            elif "error" in result_obj:
                return 0.0
            else:
                return 0.0
        except:
            return 0.0


def create_vllm_tool(
    vllm_api_url: str = "http://localhost:8000/v1/chat/completions",
    model_name: str = "default",
    topk: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 120,
    starting_molecules: str = "./dataset/origin_dict.csv",
    beam_width: int = 5,
) -> 'SingleStepRetroVLLMTool':
    """
    Factory function to create a vLLM-based single step retro tool

    Args:
        vllm_api_url: URL for vLLM API endpoint
        model_name: Model name for API calls
        topk: Number of top reactions to return
        temperature: Temperature for sampling
        max_tokens: Maximum tokens to generate
        timeout: API timeout in seconds
        starting_molecules: Path to available starting molecules file
        beam_width: Beam width for beam search (default: 5)
    """
    return SingleStepRetroVLLMTool(
        vllm_api_url=vllm_api_url,
        model_name=model_name,
        topk=topk,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        starting_molecules=starting_molecules,
        beam_width=beam_width,
    )


if __name__ == "__main__":
    # Test the tool
    tool = SingleStepRetroVLLMTool()
    print(tool.execute({"molecule": "CC(=O)Oc1ccccc1C(=O)O"}))
