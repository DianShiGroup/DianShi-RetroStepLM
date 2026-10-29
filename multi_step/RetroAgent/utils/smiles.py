from rdkit import Chem


def canonicalize_smiles(smiles):
    """Canonicalize a SMILES string using RDKit (isomeric, canonical)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
