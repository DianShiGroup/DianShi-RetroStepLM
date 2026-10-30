"""
Heuristic functions for estimating molecular synthesis difficulty.

These tools provide quantitative guidance to help the LLM agent prioritize
which molecules to expand next during retrosynthesis planning.
"""
import os
import sys
from typing import Any, Dict, List

from rdkit import Chem, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

from tools.scscore import SCScorer


def calculate_synthetic_accessibility(molecules: List[str]) -> Dict[str, Any]:
    """
    Calculate SA scores using RDKit (Ertl & Schuffenhauer, 2009).

    Args:
        molecules: List of SMILES strings to score.

    Returns:
        Dict mapping SMILES to SA score (1-10, lower = easier to synthesize).
    """
    results = {}

    invalid = []
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid.append(smiles)
            continue

        sa_score = sascorer.calculateScore(mol)
        results[smiles] = round(float(sa_score), 1)

    if invalid:
        raise ValueError(f"Invalid SMILES: {invalid[:5]}")

    return results


def calculate_scscore(scscore_model: SCScorer, molecules: List[str]) -> Dict[str, Any]:
    """
    Calculate SCScore using neural network (Coley et al., 2018).

    Trained on 12M reactions from Reaxys to learn that products are
    more complex than reactants.

    Args:
        scscore_model: Pre-loaded SCScorer model instance.
        molecules: List of SMILES strings to score.

    Returns:
        Dict mapping SMILES to SCScore (1-5, lower = easier to synthesize).
    """
    results = {}

    invalid = []
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid.append(smiles)
            continue

        _, score = scscore_model.get_score_from_smi(smiles)
        results[smiles] = round(float(score), 2)

    if invalid:
        raise ValueError(f"Invalid SMILES: {invalid[:5]}")

    return results
