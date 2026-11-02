"""Local MLP single-step retrosynthesis tool for V3/V4 checkpoints."""

import os
import pickle
import random
from pathlib import Path
from typing import Dict

import pandas as pd

from agent_r1.tool.tool_base import Tool


# Keep the local MLP runtime identical to Retro-R1's original tool.  V3/V4 is
# selected by the checkpoint path passed by the inference entrypoint; the
# remaining MLP settings are fixed here instead of being treated as API
# generation settings.
MLP_DEVICE = -1
MLP_FP_DIM = 2048
MLP_REALISTIC_FILTER = True
MLP_TEMPLATE_PATH = "./one_step_model/template_rules_1.dat"
MLP_STARTING_MOLECULES = "./dataset/origin_dict.csv"
MLP_DEFAULT_TOPK = 5
MLP_DEFAULT_CHECKPOINT = "./one_step_model/saved_rollout_state_1_2048.ckpt"


class SingleStepRetroMLPTool(Tool):
    """Run the original MLP one-step model locally."""

    def __init__(
        self,
        mlp_model_dump: str = MLP_DEFAULT_CHECKPOINT,
    ):
        name = "single_step_retro"
        description = (
            "Perform single step retrosynthesis with a local MLP model and "
            "return several candidate reactions."
        )
        parameters = {
            "type": "object",
            "properties": {
                "molecule": {
                    "type": "string",
                    "description": "The ID for the molecule to be synthesized.",
                }
            },
            "required": ["molecule"],
        }

        self.repo_root = Path(__file__).resolve().parents[2]
        self.model_path = self._resolve_path(mlp_model_dump)
        self.template_path = self._resolve_path(MLP_TEMPLATE_PATH)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MLP checkpoint not found: {self.model_path}")
        if not self.template_path.is_file():
            raise FileNotFoundError(f"MLP template file not found: {self.template_path}")

        # Import lazily so remote vLLM mode does not require PyTorch.
        from mlp_retrosyn.mlp_inference import MLPModel

        # MLPModel loads its fixed reference checkpoint from ./one_step_model.
        # Initialize it from the repository root, then restore the caller's cwd.
        previous_cwd = os.getcwd()
        try:
            os.chdir(self.repo_root)
            self.one_step = MLPModel(
                str(self.model_path),
                str(self.template_path),
                device=MLP_DEVICE,
                fp_dim=MLP_FP_DIM,
                realistic_filter=MLP_REALISTIC_FILTER,
            )
        finally:
            os.chdir(previous_cwd)

        self.use_default_mlp = self.model_path == self._resolve_path(
            MLP_DEFAULT_CHECKPOINT
        )
        self.starting_mols = self._load_starting_molecules()

        super().__init__(name, description, parameters)

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        return (self.repo_root / candidate).resolve()

    def _load_starting_molecules(self) -> set:
        candidate = self._resolve_path(MLP_STARTING_MOLECULES)
        if not candidate.exists():
            fallbacks = [
                self.repo_root / "dataset" / "origin_dict.csv",
            ]
            fallback = next((item for item in fallbacks if item.exists()), None)
            if fallback is None:
                raise FileNotFoundError(
                    f"Starting molecules file not found: {candidate}"
                )
            candidate = fallback

        if candidate.suffix == ".csv":
            return set(pd.read_csv(candidate)["mol"].tolist())
        if candidate.suffix == ".pkl":
            with candidate.open("rb") as handle:
                return set(pickle.load(handle))
        raise ValueError(f"Unsupported starting molecules file: {candidate}")

    def execute(self, args: Dict) -> Dict:
        mol = args.get("molecule", "").strip()
        topk = int(args.get("topk", MLP_DEFAULT_TOPK))
        shuffle = bool(args.get("shuffle", False))
        reaction_list = args.get("reaction_list", [])
        exist = args.get("exist", False)

        if exist:
            return {"results": reaction_list}
        if not mol:
            return {"error": "No molecule provided."}

        try:
            if mol in self.starting_mols:
                raise AssertionError(f"This molecule {mol} is already solved.")

            run_topk = 50 if self.use_default_mlp else topk
            results = self.one_step.run(mol, topk=run_topk)
            if not results:
                return {"results": []}

            reactants = []
            for reactant_smiles, score in zip(
                results["reactants"], results["scores"]
            ):
                molecules = []
                for reactant in reactant_smiles.split("."):
                    molecules.append([reactant, reactant in self.starting_mols])
                reactants.append([molecules, reactant_smiles, score])

            reactants.sort(key=lambda item: item[-1], reverse=True)
            selected = reactants[:topk]
            formatted = []
            for reaction in selected:
                formatted.append([
                    [smiles, "available" if available else "unavailable"]
                    for smiles, available in reaction[0]
                ])

            if shuffle:
                random.shuffle(formatted)
            return {"results": formatted}
        except Exception as exc:
            return {"error": str(exc)}

    def calculate_reward(self, args: Dict, result: Dict) -> float:
        return 0.0


def create_mlp_tool(
    mlp_model_dump: str = MLP_DEFAULT_CHECKPOINT,
) -> SingleStepRetroMLPTool:
    return SingleStepRetroMLPTool(mlp_model_dump=mlp_model_dump)
