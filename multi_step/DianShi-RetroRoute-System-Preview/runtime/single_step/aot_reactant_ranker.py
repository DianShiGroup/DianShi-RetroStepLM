"""AOT* stock/SCScore reward for one retrosynthesis split."""

from __future__ import annotations

import functools
import importlib.util
import math
from collections.abc import Container
from pathlib import Path

_DEFAULT_SCSCORE_ROOT = str(
    Path(__file__).resolve().parent / "aot_scscore"
)


@functools.lru_cache(maxsize=2)
def _load_scscore(scscore_root: str):
    root = Path(scscore_root)
    source = root / "scscore" / "standalone_model_numpy.py"
    weights = (
        root
        / "models"
        / "full_reaxys_model_2048bool"
        / "model.ckpt-10654.as_numpy.json.gz"
    )
    for path in (source, weights):
        if not path.is_file():
            raise FileNotFoundError(f"AOT ranking requires SCScore asset: {path}")
    spec = importlib.util.spec_from_file_location("_bench_aot_scscore", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.sigmoid = lambda value: 1.0 / (
        1.0 + math.exp(-module.np.asarray(value).item())
    )
    return module.SCScorer().restore(str(weights), FP_len=2048)


class AOTReactantRanker:
    def __init__(self, starting_mols: Container[str], scscore_root: str | None = None):
        self.starting_mols = starting_mols
        root = Path(scscore_root) if scscore_root else Path(_DEFAULT_SCSCORE_ROOT)
        self.model = _load_scscore(str(root.resolve()))
        self._molecule_score = functools.lru_cache(maxsize=50000)(self._score_molecule)

    def _score_molecule(self, smiles: str) -> float:
        score = float(self.model.get_score_from_smi(smiles)[1])
        if not math.isfinite(score) or not 0.0 <= score <= 5.0:
            raise ValueError(f"Invalid SCScore for {smiles!r}: {score}")
        return score

    def score(self, reactants: tuple[str, ...], unsolved_penalty: float = 0.0) -> float:
        scores = [
            self._molecule_score(smi)
            for smi in reactants
            if smi not in self.starting_mols
        ]
        if not scores:
            return 0.0
        total = sum(scores)
        return -(total / len(scores) + total) * (1.0 + len(scores) * unsolved_penalty)
