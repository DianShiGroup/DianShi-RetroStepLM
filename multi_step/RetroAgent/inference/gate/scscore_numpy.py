"""SCScore scorer, adapted from AOT*.

A single-step template gate has no route history, so it only needs the initial
SCScore phase of AOT*'s oracle (empty visited / dead sets). This module is the
standalone NumPy scorer from the public AOT* implementation, trimmed for
vendoring:

  * the ``six`` shim is gone (``dict.iteritems`` -> ``dict.items``), so the
    package does not carry an undeclared dependency;
  * the Python 2 ``cPickle`` weight loader is gone; only the JSON+gzip format
    used by the published checkpoints is supported;
  * the progress ``print`` became a log record;
  * unused imports and the module's ``__main__`` demo were dropped.

The arithmetic is unchanged. Weights are the published
``full_reaxys_model_2048bool`` checkpoint from connorcoley/scscore (MIT); the
gate passes ``FP_len=2048``, which is why the 1024-bit variant does not work
here.

Reference: Stratley et al., *AOT*: A dataset for retrosynthesis planning.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
from pathlib import Path
from typing import Any, List

import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

logger = logging.getLogger(__name__)

SCORE_SCALE = 5.0
FP_RAD = 2

#: Published checkpoint, vendored next to this module.
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / (
    "model.ckpt-10654.as_numpy.json.gz"
)


def _sigmoid(x):
    # The last layer leaves x with shape (1,), and NumPy 2 refuses to convert a
    # length-one array to a Python scalar. Reduce first so the score is a real
    # float, which is what the caller's float() expects.
    return 1.0 / (1.0 + math.exp(-np.asarray(x).item()))


class SCScorer:
    """AOT*'s SCScore as a feed-forward net over a Morgan bit vector."""

    def __init__(self, score_scale: float = SCORE_SCALE) -> None:
        self.vars: List[np.ndarray] = []
        self.score_scale = score_scale
        self._restored = False

    def restore(
        self,
        weight_path: str | Path,
        FP_rad: int = FP_RAD,
        FP_len: int = 2048,
    ) -> "SCScorer":
        self.FP_len = FP_len
        self.FP_rad = FP_rad
        self._load_vars(weight_path)
        logger.info("Restored SCScore variables from %s", weight_path)

        def mol_to_fp(mol):
            if mol is None:
                return np.zeros((self.FP_len,), dtype=bool)
            return np.array(
                AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.FP_rad, nBits=self.FP_len, useChirality=True
                ),
                dtype=bool,
            )

        self.mol_to_fp = mol_to_fp
        self._restored = True
        return self

    def smi_to_fp(self, smi: str):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return self.mol_to_fp(Chem.MolFromSmiles(smi))

    def apply(self, x):
        if not self._restored:
            raise ValueError("Must restore model weights!")
        # Each pair of vars is a weight and bias term.
        for i in range(0, len(self.vars), 2):
            last_layer = i == len(self.vars) - 2
            x = np.matmul(x, self.vars[i]) + self.vars[i + 1]
            if not last_layer:
                x = x * (x > 0)  # ReLU
        return 1 + (self.score_scale - 1) * _sigmoid(x)

    def get_score_from_smi(self, smi: str = "", v: bool = False):
        """Return ``(canonical_smiles, score)``; score is 0 for an empty fp."""
        if not smi:
            return ("", 0.0)
        fp = np.array(self.smi_to_fp(smi), dtype=np.float32)
        if fp.sum() == 0:
            if v:
                logger.info("Could not get fingerprint for %s", smi)
            cur_score: Any = 0.0
        else:
            cur_score = self.apply(fp)
            if v:
                logger.info("Score: %s", cur_score)
        mol = Chem.MolFromSmiles(smi)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True) if mol else ""
        return (canonical, cur_score)

    def _load_vars(self, weight_path: str | Path) -> None:
        path = Path(weight_path)
        if not path.is_file():
            raise FileNotFoundError("SCScore weights not found: %s" % path)
        if path.suffix != ".gz":
            raise ValueError(
                "SCScore weights must be the JSON+gzip checkpoint, got: %s" % path
            )
        with gzip.GzipFile(str(path), "r") as handle:
            payload = json.loads(handle.read().decode("utf-8"))
        self.vars = [np.array(layer) for layer in payload]


def load_scorer(weights: str | Path | None = None, FP_len: int = 2048) -> SCScorer:
    """Load the vendored checkpoint (or a caller-supplied one)."""
    path = Path(weights) if weights else DEFAULT_WEIGHTS
    return SCScorer().restore(path, FP_len=FP_len)


__all__ = ["SCScorer", "load_scorer", "DEFAULT_WEIGHTS", "SCORE_SCALE", "FP_RAD"]