"""Planner prompt construction and SMILES canonicalization."""

from typing import Dict, List, Tuple

from rdkit import Chem


INSTRUCTION_FOLLOWING = """You are a professional organic chemist, skilled at finding pathways to synthesize novel molecules. Given a target molecule, your task is to find a pathway where the target can be eventually synthesized using ingredient molecules, and each step of the pathway is a viable chemical reaction.
You can conduct 2 actions:
1 'Call'. Select an unsolved molecule and I'll provide you with several reactions to synthesize it. Note that the reactions may be incorrect and the reactants are marked as 'available' or 'unavailable'. The unavailable molecules have to be solved further.
2 'Select'. After calling the one-step model, you can use this action to select a viable reaction. If the reaction has unavailable reactants, please make sure they are easier to synthesize.
The maximum number of reaction steps you can use is {n}.

Strategy tips:
- Always pick the unsolved molecule that looks hardest to make.
- Prefer reactions where most reactants are already available.
- Think step-by-step before every tool call.
"""

QUESTION_FOLLOWING = """The target molecule is {mol}. Now start to search for the pathway.
Molecule state 0: Unsolved molecules:
Molecule 0-0: {mol}
Now, 0 steps are used.
Please select the molecule with the highest synthesis difficulty from the unsolved molecules. Use the tool 'single_step_retro' to synthesize it."""


def canonicalize_smiles_clear_map(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def build_retro_prompt(
    smiles: str, maxstep: int = 30, canonicalize: bool = True
) -> Tuple[str, str]:
    source = smiles.strip()
    if not source:
        raise ValueError("SMILES must not be empty")
    target = canonicalize_smiles_clear_map(source) if canonicalize else source
    if not target:
        raise ValueError(f"Invalid SMILES: {source}")
    return target, INSTRUCTION_FOLLOWING.format(n=maxstep) + QUESTION_FOLLOWING.format(mol=target)


def build_txt_prompt(target: str, maxstep: int) -> List[Dict[str, str]]:
    """Return the legacy user-message shape used by the current scripts."""
    _, content = build_retro_prompt(target, maxstep=maxstep, canonicalize=False)
    return [{"role": "user", "content": content}]
