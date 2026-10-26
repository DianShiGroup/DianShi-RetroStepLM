import functools
import logging
import os
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

SQLITE_STOCK_SUFFIXES = (".sqlite", ".sqlite3", ".db")


def _resolve_starting_molecules_path(path: str) -> Optional[str]:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    configured_paths = [path]

    if not os.path.isabs(path):
        configured_paths.append(os.path.join(repo_root, path))
        configured_paths.append(os.path.join(os.getcwd(), path))

    for candidate in configured_paths:
        if os.path.exists(candidate):
            return candidate

    fallback_paths = [
        os.path.join(repo_root, "data", "origin_dict.csv"),
        os.path.join(os.getcwd(), "data", "origin_dict.csv"),
    ]
    for fallback_path in fallback_paths:
        if os.path.exists(fallback_path):
            logger.warning(
                "Configured starting molecules file was not found: %s; "
                "falling back to default stock file: %s",
                path,
                fallback_path,
            )
            return fallback_path

    return None


def _is_sqlite_stock_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SQLITE_STOCK_SUFFIXES


@functools.lru_cache(maxsize=8)
def _load_starting_molecules_cached(resolved_path: str) -> frozenset[str]:
    suffix = os.path.splitext(resolved_path)[1].lower()

    if suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(resolved_path)
        raw_mols = df["mol"].dropna().astype(str).str.strip().tolist()
    elif suffix == ".txt":
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_mols = [
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]
    elif suffix == ".pkl":
        import pickle

        with open(resolved_path, "rb") as f:
            raw_mols = [str(value).strip() for value in pickle.load(f)]
    else:
        if _is_sqlite_stock_path(resolved_path):
            return frozenset()
        logger.warning("Unsupported starting molecules format: %s", resolved_path)
        return frozenset()

    return frozenset(raw_mols)


def preload_starting_molecules(path: str) -> bool:
    resolved_path = _resolve_starting_molecules_path(path)
    if resolved_path is None:
        logger.warning(
            "Starting molecules preload skipped because file was not found: %s",
            path,
        )
        return False

    if _is_sqlite_stock_path(resolved_path):
        try:
            with sqlite3.connect(resolved_path) as conn:
                conn.execute("SELECT 1 FROM stocks LIMIT 1").fetchone()
            logger.info("Using stock sqlite backend %s", resolved_path)
        except Exception as exc:
            logger.error("Failed to preload sqlite stock %s: %s", resolved_path, exc)
            return False
        return True

    logger.info("[1/2]: Loading Commercial Stocks from %s ...", resolved_path)
    _load_starting_molecules_cached(resolved_path)
    return True
