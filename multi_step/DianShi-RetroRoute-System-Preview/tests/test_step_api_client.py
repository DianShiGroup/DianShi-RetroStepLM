from __future__ import annotations

import sqlite3

from runtime import step_api_client
from runtime.step_api_client import StepAPIClient


def test_step_api_client_reuses_cached_starting_molecules(monkeypatch):
    cached = frozenset({"CCO", "CCN"})

    monkeypatch.setattr(
        step_api_client,
        "_resolve_starting_molecules_path",
        lambda path: "/tmp/stock.pkl",
    )
    monkeypatch.setattr(
        step_api_client,
        "_load_starting_molecules_cached",
        lambda resolved_path: cached,
    )

    client = StepAPIClient(base_url="http://example.test", starting_molecules="stock.pkl")

    assert client.starting_mols is cached


def test_step_api_client_sqlite_starting_molecules(tmp_path):
    db_path = tmp_path / "stocks.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE stocks(smiles TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO stocks(smiles) VALUES (?)",
            [("CCO",), ("CCN",)],
        )

    client = StepAPIClient(base_url="http://example.test", starting_molecules=str(db_path))

    assert len(client.starting_mols) == 2
    assert "CCO" in client.starting_mols
    assert "CCC" not in client.starting_mols
