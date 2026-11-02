"""
DianShi-RetroStepLM-Direct single-step tool: an OpenAI-compatible HTTP step model with the
rdchiral/AOT* template gate, no retrieval.
"""

import os
import re
import logging
import time
from typing import Dict, Any, List, Optional
from agent_r1.tool.tool_base import Tool
import random

from inference.support.prompt_sections import build_direct_messages, load_prompt_sections
from inference.support.smiles_canonical import canonicalize_smiles_clear_map
from inference.gate.rdchiral_template_filter import build_rdchiral_template_filter
from inference.support.smiles_atomwise import restore_answer_spaces, transform_messages
from inference.stock import DEFAULT_STOCK_PATH, BuildingBlockChecker

logger = logging.getLogger(__name__)
ANSWER_RE = re.compile(r"<answer\s*>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


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


class DianShiDirectStepRetroTool(Tool):
    """
    Tool for performing Single step retrosynthesis using vLLM API
    """

    def __init__(
        self,
        step_api_url: str = "http://localhost:8000/v1/chat/completions",
        step_api_urls: Optional[List[str]] = None,
        prompt_template: str = None,
        model_name: str = "default",
        topk: int = 10,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 120,
        starting_molecules: str = str(DEFAULT_STOCK_PATH),
        beam_width: int = 5,
        api_key: Optional[str] = None,
        atomwise: bool = False,
        prompt_path_no_rag: Optional[str] = None,
        template_filter: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the vLLM-based single step retro tool

        Args:
            step_api_url: URL for a OpenAI-compatible endpoint
            step_api_urls: Optional list of OpenAI-compatible endpoints
            prompt_template: Path to prompt template file
            model_name: Name of the model for API calls
            topk: Number of reactions to return
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            timeout: Timeout for API calls in seconds
            starting_molecules: Path to file containing available starting molecules
            beam_width: Beam width for beam search (generates beam_width candidates, returns top-k)
            api_key: Optional API key for the vLLM endpoint
        """
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

        raw_api_urls = step_api_urls or [step_api_url]
        if isinstance(raw_api_urls, str):
            raw_api_urls = [raw_api_urls]
        self.step_api_urls = [url.strip() for url in raw_api_urls if url and url.strip()]
        if not self.step_api_urls:
            raise ValueError("At least one vLLM API URL is required")
        self.step_api_url = self.step_api_urls[0]
        self.model_name = model_name
        self.topk = topk
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.beam_width = beam_width
        self.api_key = api_key
        self.atomwise = atomwise

        prompt_path = prompt_path_no_rag
        if prompt_path:
            prompt_path = self._resolve_local_path(prompt_path)
            if not os.path.isfile(prompt_path):
                raise FileNotFoundError(f"Single-step prompt not found: {prompt_path}")
            self.prompt_sections = load_prompt_sections(prompt_path)
        else:
            self.prompt_sections = None

        # Load starting molecules (available building blocks)
        self.starting_mols = self._load_starting_molecules(starting_molecules)
        self.template_filter = build_rdchiral_template_filter(
            template_filter, starting_mols=self.starting_mols
        )

        super().__init__(name, description, parameters)

    @staticmethod
    def _resolve_local_path(path: str) -> str:
        if os.path.isabs(path):
            return path
        candidates = [
            os.path.join(os.getcwd(), path),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return candidates[0]

    def _load_starting_molecules(self, path: str):
        """Load starting molecules from sqlite (preferred) or csv/pkl."""
        if not os.path.exists(path):
            alt_sqlite = str(DEFAULT_STOCK_PATH)
            alt_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "origin_dict.csv")
            if os.path.exists(alt_sqlite):
                path = alt_sqlite
            elif os.path.exists(alt_csv):
                path = alt_csv
            else:
                cwd_path = os.path.join(os.getcwd(), "dataset", "origin_dict.csv")
                if os.path.exists(cwd_path):
                    path = cwd_path
                else:
                    logger.warning(
                        "Starting molecules file not found at %s. Building blocks will be marked as unavailable.",
                        path,
                    )
                    return set()

        lowered = path.lower()
        if lowered.endswith(".sqlite") or lowered.endswith(".db"):
            return BuildingBlockChecker(path)
        import pandas as pd
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            return set(df["mol"].tolist())
        if path.endswith(".pkl"):
            import pickle
            with open(path, "rb") as f:
                return set(pickle.load(f))
        return set()

    def _build_messages(
        self, product_smiles: str
    ) -> tuple[str, List[Dict[str, str]], List[Dict[str, Any]]]:
        canonical_product = canonicalize_smiles_clear_map(product_smiles)
        if not canonical_product:
            raise ValueError(f"Invalid product SMILES: {product_smiles}")

        if self.prompt_sections is not None:
            messages = build_direct_messages(
                product_smiles=canonical_product,
                prompt_sections=self.prompt_sections,
            )
        else:
            prompt = self.prompt_template.replace("{products}", canonical_product)
            messages = [
                {
                    "role": "system",
                    "content": prompt.split("---USER---")[0]
                    .replace("---SYSTEM---", "")
                    .strip(),
                },
                {
                    "role": "user",
                    "content": prompt.split("---USER---")[1]
                    .split("---ASSISTANT---")[0]
                    .strip(),
                },
            ]

        if self.atomwise:
            messages, transformed_spans = transform_messages(messages)
            logger.info(
                "Atom-wise SMILES formatting enabled | transformed_spans=%d",
                transformed_spans,
            )

        return canonical_product, messages

    @staticmethod
    def _extract_predictions(result: Dict[str, Any], atomwise: bool) -> List[Dict[str, Any]]:
        predictions = []
        for rank, choice in enumerate(result.get("choices", []), start=1):
            content = choice.get("message", {}).get("content", "") or ""
            if atomwise:
                content = restore_answer_spaces(content).text

            match = ANSWER_RE.search(content)
            answer = match.group(1).strip() if match else ""
            predictions.append(
                {
                    "rank": rank,
                    "output_smiles": answer,
                    "parsed": [
                        part.strip() for part in answer.split(".") if part.strip()
                    ],
                }
            )
        return predictions

    def _call_api(self, product_smiles: str) -> List[Dict[str, Any]]:
        """Call the single-step endpoint and return the parsed candidates."""
        import requests

        canonical_product, messages = self._build_messages(product_smiles)
        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_beam_search": True,
            "best_of": self.beam_width,
            "n": self.beam_width,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Optional[Exception] = None
        for attempt in range(4):
            api_url = random.choice(self.step_api_urls)
            try:
                logger.info(
                    "Calling vLLM API %s | molecule=%s | attempt=%d",
                    api_url,
                    canonical_product,
                    attempt + 1,
                )
                response = requests.post(
                    api_url,
                    json=request_payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()
                predictions = self._extract_predictions(result, self.atomwise)
                if any(item["parsed"] for item in predictions) or attempt == 3:
                    return predictions
                last_error = ValueError("vLLM response contained no <answer> result")
            except requests.exceptions.Timeout as exc:
                last_error = TimeoutError(
                    f"vLLM API timeout after {self.timeout}s"
                )
                logger.warning("Single-step timeout: %s", exc)
            except requests.exceptions.RequestException as exc:
                last_error = ConnectionError(f"vLLM API error: {exc}")
                logger.warning("Single-step request failed: %s", exc)
            except ValueError as exc:
                last_error = ValueError(f"Invalid vLLM response: {exc}")
                logger.warning("Single-step response parsing failed: %s", exc)

            if attempt < 3:
                time.sleep(min(2 ** attempt, 10))

        assert last_error is not None
        raise last_error

    @staticmethod
    def _predictions_to_tuples(predictions: List[Dict[str, Any]]) -> List[tuple]:
        results = []
        for item in predictions:
            if not isinstance(item, dict):
                continue
            parsed = item.get("parsed")
            if isinstance(parsed, list) and parsed:
                smiles = ".".join(
                    str(part).strip() for part in parsed if str(part).strip()
                )
            else:
                smiles = str(item.get("output_smiles") or "").strip()
            if smiles:
                results.append((smiles, None))
        return results

    def _predict_reactants(self, canonical_mol: str) -> List[tuple]:
        return self._predictions_to_tuples(self._call_api(canonical_mol))

    def _parse_reactants_to_result(self, reactant_smiles: str, score: float = None) -> List[List]:
        """
        Parse reactant SMILES string to the result format

        Args:
            reactant_smiles: Reactant SMILES string (can contain multiple separated by '.')
            score: Score for this reaction

        Returns:
            List of [[reactant_smiles, availability], ...] - flattened format compatible with MLP model
        """
        reactant_list = [
            part.strip() for part in (reactant_smiles or "").split(".") if part.strip()
        ]
        if not reactant_list:
            return []

        canonical_reactants = []
        for reactant in reactant_list:
            canonical = canonicalize_smiles_clear_map(reactant)
            if not canonical:
                return []
            canonical_reactants.append(canonical)

        reactant_list_exist = []
        for reactant in canonical_reactants:
            is_available = reactant in self.starting_mols
            reactant_list_exist.append([reactant, 'available' if is_available else 'unavailable'])

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

        canonical_mol = canonicalize_smiles_clear_map(mol)
        if not canonical_mol:
            return {"error": f"Invalid molecule SMILES: {mol}"}

        try:
            # Check if it's available molecule
            if canonical_mol in self.starting_mols:
                return {"results": []}

            reactant_predictions = self._predict_reactants(canonical_mol)

            if not reactant_predictions:
                return {"results": []}

            parsed = []
            seen_reactions = set()
            for pred in reactant_predictions:
                reactant_smiles, logprobs = pred
                result = self._parse_reactants_to_result(reactant_smiles, logprobs)
                if not result:
                    continue
                reaction_key = tuple(sorted(item[0] for item in result))
                if reaction_key in seen_reactions:
                    continue
                seen_reactions.add(reaction_key)
                parsed.append(result)

            if self.template_filter is not None:
                filtered = self.template_filter.filter_candidates(
                    product=canonical_mol,
                    candidates=[[item[0] for item in row] for row in parsed],
                )
                reactants_sorted = [
                    [
                        [
                            smiles,
                            "available"
                            if smiles in self.starting_mols
                            else "unavailable",
                        ]
                        for smiles in result.reactants
                    ]
                    for result in filtered
                    if result.reactants
                ]
            else:
                reactants_sorted = parsed

            if shuffle:
                random.shuffle(reactants_sorted)

            return {"results": reactants_sorted[:topk]}

        except Exception as e:
            logger.error(
                "Error in vLLM single step retro for molecule %s: %s",
                canonical_mol,
                e,
                exc_info=True,
            )
            raise ValueError(
                f"Error in vLLM single step retro for molecule {canonical_mol}: {e}"
            ) from e

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


def create_dianshi_direct_tool(
    step_api_url: str = "http://localhost:8000/v1/chat/completions",
    step_api_urls: Optional[List[str]] = None,
    model_name: str = "default",
    topk: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout: int = 120,
    starting_molecules: str = str(DEFAULT_STOCK_PATH),
    beam_width: int = 5,
    api_key: Optional[str] = None,
    atomwise: bool = False,
    prompt_path_no_rag: Optional[str] = None,
    template_filter: Optional[Dict[str, Any]] = None,
) -> 'DianShiDirectStepRetroTool':
    """
    Factory function to create a vLLM-based single step retro tool

    Args:
        step_api_url: URL for a OpenAI-compatible endpoint
        step_api_urls: Optional list of OpenAI-compatible endpoints
        model_name: Model name for API calls
        topk: Number of top reactions to return
        temperature: Temperature for sampling
        max_tokens: Maximum tokens to generate
        timeout: API timeout in seconds
        starting_molecules: Path to available starting molecules file
        beam_width: Beam width for beam search (default: 5)
        api_key: Optional API key for the vLLM endpoint
    """
    return DianShiDirectStepRetroTool(
        step_api_url=step_api_url,
        step_api_urls=step_api_urls,
        model_name=model_name,
        topk=topk,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        starting_molecules=starting_molecules,
        beam_width=beam_width,
        api_key=api_key,
        atomwise=atomwise,
        prompt_path_no_rag=prompt_path_no_rag,
        template_filter=template_filter,
    )


if __name__ == "__main__":
    # Test the tool
    tool = DianShiDirectStepRetroTool()
    print(tool.execute({"molecule": "CC(=O)Oc1ccccc1C(=O)O"}))
