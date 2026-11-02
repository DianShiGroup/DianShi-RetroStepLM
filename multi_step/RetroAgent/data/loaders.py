import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np

from utils.smiles import canonicalize_smiles

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROCESSED_DIR = Path("datasets/processed")
BUILDING_BLOCKS_PATH = PROCESSED_DIR / "building_blocks.parquet"
RETRO_MODEL_PATH = "datasets/model_retro.pt"
RETRO_TEMPLATES_PATH = "datasets/idx2template_retro.json"
SCSCORE_MODEL_PATH = "datasets/scscore/model.ckpt-10654.as_numpy.json.gz"

# USPTO-Full training route data paths
TRAIN_ROUTES_DIR = Path("datasets/train_routes")
TRAIN_ROUTES = TRAIN_ROUTES_DIR / "routes_train.pkl"
VAL_ROUTES = TRAIN_ROUTES_DIR / "routes_val.pkl"

# Evaluation target dataset paths
TARGET_DATASETS = {
    "uspto-190": Path("datasets/uspto_190_targets.txt"),
    "chembl-1000": Path("datasets/chembl_1000.pkl"),
}


def load_targets_pkl(path):
    """
    Load target molecules from a pickle file.

    Expects a list of [SMILES] single-element lists. Each SMILES is
    canonicalized and returned as a target dict.

    Args:
        path: Path to the pickle file.

    Returns:
        List of target dictionaries with 'target' key.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    all_targets = []
    for entry in data:
        smiles = entry[0] if isinstance(entry, list) else entry
        canon = canonicalize_smiles(smiles)
        if canon:
            all_targets.append({"target": canon})
    return all_targets


def load_targets(dataset_name="uspto-190"):
    """
    Load target molecules from a file.

    Args:
        dataset_name: Name of a predefined evaluation dataset ('uspto-190', 'chembl-1000').

    Returns:
        List of target dictionaries with 'target' key (and 'start_material' for text files).
    """
    path = TARGET_DATASETS[dataset_name]
    logger.info(f"Loading targets {dataset_name} from {path}...")

    if path.suffix == ".pkl":
        all_targets = load_targets_pkl(path)
    else:
        all_targets = []
        pattern = r"'([^']+)',\s*'([^']+)'"
        with open(path, "r") as f:
            for line in f:
                match = re.search(pattern, line.strip())
                if match:
                    target, start_material = match.groups()
                    all_targets.append(
                        {"target": target, "start_material": start_material}
                    )

    logger.info(f"Loaded {len(all_targets)} targets")
    return all_targets


def load_targets_txt(path):
    """Load one product SMILES per non-empty line from a text file."""
    path = Path(path)
    all_targets = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            smiles = line.strip()
            if not smiles:
                continue
            canonical = canonicalize_smiles(smiles)
            if not canonical:
                raise ValueError(
                    f"Invalid SMILES in target file {path} at line {line_number}: {smiles}"
                )
            all_targets.append({"target": canonical})
    logger.info(f"Loaded {len(all_targets)} targets from {path}")
    return all_targets


def load_target_data(
    target_smiles=None,
    targets="uspto-full",
    num_targets=None,
    train_depth_min=None,
    train_depth_max=None,
):
    """Load and normalize target data from any dataset.

    Args:
        target_smiles: Single target SMILES override.
        targets: Dataset name (uspto-full, uspto-190, chembl-1000).
        num_targets: Max number of targets to return. None for all.
        train_depth_min: Min route depth for uspto-full training targets.
        train_depth_max: Max route depth for uspto-full training targets.

    Returns:
        List of dicts with keys: target, reference_route, depth.
    """
    if target_smiles:
        return [{"target": target_smiles, "reference_route": None, "depth": None}]

    targets_path = Path(targets).expanduser()
    if targets_path.is_file():
        raw = load_targets_txt(targets_path)
    elif targets == "uspto-full":
        raw = load_train_routes(
            depth_min=train_depth_min,
            depth_max=train_depth_max,
        )
    else:
        raw = load_targets(dataset_name=targets)

    normalized = []
    for item in raw[:num_targets]:
        if isinstance(item, dict):
            normalized.append(
                {
                    "target": item.get("target", ""),
                    "reference_route": item.get("reference_route"),
                    "depth": item.get("depth"),
                }
            )
        else:
            normalized.append(
                {
                    "target": item,
                    "reference_route": None,
                    "depth": None,
                }
            )
    return normalized


def load_train_routes(depth_min=None, depth_max=None):
    """
    Sample targets from the USPTO-Full route datasets (train + val) within a depth range.

    Depth is the length of the stored reference route. Targets are canonicalized SMILES.
    """
    with open(TRAIN_ROUTES, "rb") as f:
        train_routes = pickle.load(f)
    with open(VAL_ROUTES, "rb") as f:
        val_routes = pickle.load(f)

    routes = list(train_routes) + list(val_routes)
    np.random.shuffle(routes)

    excluded_targets = set()
    for _, path in TARGET_DATASETS.items():
        if path.suffix == ".pkl":
            for t in load_targets_pkl(path):
                excluded_targets.add(t["target"])
        else:
            with open(path, "r") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    if text[0] in "([" and "," in text:
                        text = text.split(",", 1)[0].strip()
                        text = text.lstrip("('\"").rstrip("')\"")
                    canon = canonicalize_smiles(text)
                    if canon:
                        excluded_targets.add(canon)
    if excluded_targets:
        logger.info(
            f"Excluding {len(excluded_targets)} targets present in test datasets"
        )

    selected = []
    seen_smiles = set()

    for route in routes:
        depth = len(route)
        if depth_min is not None and depth < depth_min:
            continue
        if depth_max is not None and depth > depth_max:
            continue
        target_smiles = route[0].split(">>")[0]
        canon = canonicalize_smiles(target_smiles)
        if not canon or canon in seen_smiles or canon in excluded_targets:
            continue
        seen_smiles.add(canon)
        reference_route = []
        for rxn in route:
            product, reactants = rxn.split(">>")
            reference_route.append(
                {
                    "product": product,
                    "reactants": reactants.split(".") if reactants else [],
                    "reaction": rxn,
                }
            )
        selected.append(
            {
                "target": canon,
                "depth": depth,
                "reference_route": reference_route,
                "reference_length": len(reference_route),
            }
        )

    if depth_min is None and depth_max is None:
        logger.info(f"Loaded {len(selected)} post-training targets (no depth filter).")
    else:
        logger.info(
            f"Loaded {len(selected)} post-training targets (depth {depth_min}-{depth_max})."
        )
    return selected
