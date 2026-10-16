"""AOT*'s stock/SCScore reward for a single retrosynthesis step.

A single-step gate has no route history: use the initial SCScore phase of
AOT*'s oracle, with empty visited/dead molecule sets. The scorer itself is
vendored next to this module (see scscore_numpy.py), so nothing outside this
repository is needed at runtime.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from collections.abc import Container

from inference.gate.scscore_numpy import DEFAULT_WEIGHTS, load_scorer


class AOTReactantRanker:
    def __init__(self, starting_mols: Container[str], scscore_weights: str | None = None):
        self.starting_mols = starting_mols
        weights = Path(scscore_weights) if scscore_weights else Path(DEFAULT_WEIGHTS)
        if not weights.is_file():
            raise FileNotFoundError(
                "AOT* ranking needs the SCScore checkpoint; set scscore_weights "
                f"in the config or export SCSCORE_WEIGHTS (looked in {weights})"
            )
        self.model = load_scorer(weights)
        # Per-instance bounded caches: stock and model do not change during a run.
        self._molecule_score = functools.lru_cache(maxsize=50000)(self._score_molecule)

    def _score_molecule(self, smiles: str) -> float:
        score = float(self.model.get_score_from_smi(smiles)[1])
        # AOT* keeps a zero score for molecules whose fingerprint is empty.
        if not math.isfinite(score) or not 0.0 <= score <= 5.0:
            raise ValueError(f"Invalid SCScore for {smiles!r}: {score}")
        return score

    def score(self, reactants: tuple[str, ...], unsolved_penalty: float = 0.0) -> float:
        """Higher is better; all-stock is 0, otherwise -(mean + sum).

        AOT*'s blurry reranker also multiplies by (1 + unsolved_count * gamma).
        Its direct verifier uses the unadjusted reward.
        """
        scores = [self._molecule_score(smi) for smi in reactants if smi not in self.starting_mols]
        if not scores:
            return 0.0
        total = sum(scores)
        return -(total / len(scores) + total) * (1.0 + len(scores) * unsolved_penalty)
