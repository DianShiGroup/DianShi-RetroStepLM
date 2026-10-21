import functools
import sqlite3
import threading
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

STOCKS_PATH = (
    Path(__file__).resolve().parent.parent / "dataset" / "emolecules_standardized.sqlite"
)


def canonicalize_stock_smiles(smiles: str):
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
    """Exact canonical-SMILES lookup against the on-disk eMolecules stock.

    Query strings are canonicalized with RDKit before lookup so callers do not
    need the same SMILES writing as the database.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else STOCKS_PATH
        if not self.path.is_file():
            raise FileNotFoundError("building-block sqlite not found: %s" % self.path)
        self._uri = f"file:{self.path.as_posix()}?mode=ro"
        self._connections = threading.local()
        self._len = None
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

    def check(self, smiles):
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
