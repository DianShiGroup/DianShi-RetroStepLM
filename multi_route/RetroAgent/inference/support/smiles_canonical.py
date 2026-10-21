"""SMILES canonicalisation helpers shared by the step backend and the gate.

Both the backend and the template gate need reactant/product strings
normalised the same way before they can be compared or matched, so the two
functions live here rather than inside either of them.
"""

from __future__ import annotations

from rdkit import Chem


def canonicalize_smiles_clear_map(smiles: str) -> str:
    value = str(smiles or "").strip()
    if not value:
        return ""
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def canonicalize_reactants(reactants: str) -> str:
    parts = [part.strip() for part in str(reactants or "").split(".") if part.strip()]
    if not parts:
        return ""
    canonical_parts = []
    for part in parts:
        canonical = canonicalize_smiles_clear_map(part)
        if not canonical:
            return ""
        canonical_parts.append(canonical)
    canonical_parts.sort()
    return ".".join(canonical_parts)
