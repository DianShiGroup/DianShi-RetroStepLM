"""
SCScore (Synthetic Complexity Score) implementation.

Based on Coley et al. (2018) - trained on 12M reactions from Reaxys.
Standalone NumPy implementation without TensorFlow dependency.
"""
import gzip
import json
import logging
import os
from typing import Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from data.loaders import SCSCORE_MODEL_PATH

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SCORE_SCALE = 5.0
FP_RAD = 2
FP_LEN = 1024


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SCScorer:
    """
    SCScore model for predicting synthetic complexity.

    Scores range from 1 (easy) to 5 (hard).
    """

    def __init__(
        self, weight_path: str = SCSCORE_MODEL_PATH, score_scale: float = SCORE_SCALE
    ):
        self.score_scale = score_scale
        self.fp_len = FP_LEN
        self.fp_rad = FP_RAD

        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"SCScore model not found at {weight_path}. "
                "Please download from https://github.com/connorcoley/scscore"
            )

        if weight_path.endswith(".json.gz"):
            with gzip.GzipFile(weight_path, "r") as fin:
                self.vars = [
                    np.array(x) for x in json.loads(fin.read().decode("utf-8"))
                ]
        elif weight_path.endswith(".json"):
            with open(weight_path, "r") as fin:
                self.vars = [np.array(x) for x in json.load(fin)]
        else:
            raise ValueError(f"Unsupported weight format: {weight_path}")

        logger.info(f"Loaded SCScore model from {weight_path}")

    def mol_to_fp(self, mol) -> np.ndarray:
        """Convert RDKit mol to fingerprint."""
        if mol is None:
            return np.zeros((self.fp_len,), dtype=np.float32)
        return np.array(
            AllChem.GetMorganFingerprintAsBitVect(
                mol, self.fp_rad, nBits=self.fp_len, useChirality=True
            ),
            dtype=np.float32,
        )

    def smi_to_fp(self, smi: str) -> np.ndarray:
        """Convert SMILES to fingerprint."""
        if not smi:
            return np.zeros((self.fp_len,), dtype=np.float32)
        return self.mol_to_fp(Chem.MolFromSmiles(smi))

    def apply(self, x: np.ndarray) -> float:
        """Apply MLP to fingerprint."""
        for i in range(0, len(self.vars), 2):
            last_layer = i == len(self.vars) - 2
            W = self.vars[i]
            b = self.vars[i + 1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0)  # ReLU
        x = 1 + (self.score_scale - 1) * sigmoid(x)
        return float(np.squeeze(x))

    def get_score_from_smi(self, smi: str) -> Tuple[str, float]:
        """Get SCScore for a SMILES string."""
        if not smi:
            return ("", 0.0)
        fp = self.smi_to_fp(smi)
        if fp.sum() == 0:
            return (smi, 0.0)
        score = self.apply(fp)
        # Canonicalize SMILES
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        return (smi, score)
