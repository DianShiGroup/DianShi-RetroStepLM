"""eMolecules sqlite stock: canonicalize with RDKit, then exact SMILES lookup."""

from __future__ import annotations

import functools
import os
import sqlite3
import threading
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# The building-block catalogue is not vendored (it is ~1 GB and derived from
# eMolecules). Resolve it as: explicit config value > STARTING_MOLECULES >
# this repository's dataset/ directory; see the README for where to get it.
DEFAULT_STOCK_PATH = Path(
    os.environ.get("STARTING_MOLECULES")
    or Path(__file__).resolve().parents[1] / "dataset" / "emolecules_standardized.sqlite"
)


def canonicalize_stock_smiles(smiles: str) -> str | None:
    value = str(smiles or "").strip()
    if not value:
        return None
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


class BuildingBlockChecker:
    """Exact canonical-SMILES lookup against the on-disk eMolecules stock."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_STOCK_PATH
        if not self.path.is_file():
            raise FileNotFoundError("building-block sqlite not found: %s" % self.path)
        self._uri = f"file:{self.path.as_posix()}?mode=ro"
        self._connections = threading.local()
        self._len: int | None = None
        self._lookup = functools.lru_cache(maxsize=100000)(self._db_contains)

    def _connection(self):
        connection = getattr(self._connections, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self._uri, uri=True)
            self._connections.connection = connection
        return connection

    def _db_contains(self, smiles: str) -> bool:
        found = self._connection().execute(
            "SELECT 1 FROM stocks WHERE smiles = ? LIMIT 1", (smiles,)
        ).fetchone()
        return found is not None

    def check(self, smiles: str) -> tuple[bool, str | None]:
        canon = canonicalize_stock_smiles(smiles)
        if not canon:
            return False, None
        return self._lookup(canon), canon

    def __contains__(self, smiles: object) -> bool:
        if not isinstance(smiles, str):
            return False
        ok, _ = self.check(smiles)
        return ok

    def __len__(self) -> int:
        if self._len is None:
            self._len = int(
                self._connection().execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
            )
        return self._len
