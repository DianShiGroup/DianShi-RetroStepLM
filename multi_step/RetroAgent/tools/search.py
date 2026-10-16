"""
Agent tools for stepwise retrosynthesis planning.

This module provides tool functions for retrosynthesis planning agents.
All functions are pure Python with docstrings and type hints for ADK compatibility.
"""

from typing import Any, Dict, List

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from data.building_blocks import BuildingBlockChecker


def search_building_blocks(
    bb_checker: BuildingBlockChecker, molecules: List[str]
) -> Dict[str, Any]:
    """
    Search for building blocks matching the given molecules.

    This tool caches its results. Once a molecule is checked, it's remembered
    for the rest of the execution.

    Args:
        bb_checker: Building block checker with cached lookup.
        molecules: List of SMILES strings to search for.

    Returns:
        Dictionary with results for each molecule including availability status.
    """
    results = {}
    found = 0
    invalid_smiles = 0
    missing = 0

    for molecule in molecules:
        ok, canon = bb_checker.check(molecule)
        if ok:
            results[molecule] = {
                "building_block": True,
                "canonical_SMILES": canon,
            }
            found += 1
        else:
            if canon is None:
                results[molecule] = {"building_block": False, "error": "invalid_smiles"}
                invalid_smiles += 1
            else:
                results[molecule] = {"building_block": False}
                missing += 1

    return {
        "results": results,
        "summary": {
            "total_molecules": len(molecules),
            "found_building_blocks": found,
            "missing_building_blocks": missing,
            "invalid_smiles": invalid_smiles,
        },
    }


def get_molecule_info(molecules: List[str]) -> Dict[str, Any]:
    """
    Get basic molecular information for a list of SMILES strings.

    Args:
        molecules: List of SMILES strings to analyze.

    Returns:
        Dictionary mapping each SMILES to its molecular properties.
    """
    results = {}
    invalid = []
    for smiles in molecules:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid.append(smiles)
            continue
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_atoms = mol.GetNumAtoms()
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        results[smiles] = {
            "canonical_SMILES": Chem.MolToSmiles(mol),
            "molecular_weight": round(float(mw), 1),
            "logp": round(float(logp), 2),
            "num_atoms": int(num_atoms),
            "num_rings": int(num_rings),
        }
    if invalid:
        raise ValueError(f"Invalid SMILES: {invalid[:5]}")
    return results
