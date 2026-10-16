"""
Basic SMILES hygiene functions for Worker pipeline.

These are pure data-cleansing operations — no chemical judgment.
"""

import logging

logger = logging.getLogger(__name__)


def canonicalize_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        for atom in mol.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return ""


def clean_candidates(raw_candidates: list) -> list:
    seen = set()
    cleaned = []
    total = len(raw_candidates)
    invalid_count = 0
    dup_count = 0

    for reaction in raw_candidates:
        if isinstance(reaction, list):
            reactant_smiles = [r[0] for r in reaction if isinstance(r, (list, tuple))]
        elif isinstance(reaction, str):
            reactant_smiles = [r.strip() for r in reaction.split(".") if r.strip()]
        else:
            invalid_count += 1
            continue

        valid = True
        canonicals = []
        for s in reactant_smiles:
            c = canonicalize_smiles(s)
            if not c:
                valid = False
                break
            canonicals.append(c)

        if not valid or len(canonicals) == 0:
            invalid_count += 1
            continue

        key = ".".join(sorted(canonicals))
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)

        clean_reaction = [
            [
                canonicals[i],
                reaction[i][1]
                if isinstance(reaction[i], (list, tuple))
                else "unavailable",
            ]
            for i in range(len(canonicals))
        ]
        cleaned.append(clean_reaction)

    after_invalid = total - invalid_count
    after_dedup = len(cleaned)
    logger.info(
        "SMILES validation: input=%d invalid=%d remaining=%d",
        total,
        invalid_count,
        after_invalid,
    )
    logger.info(
        "Deduplication: input=%d filtered=%d remaining=%d",
        after_invalid,
        dup_count,
        after_dedup,
    )

    return cleaned
