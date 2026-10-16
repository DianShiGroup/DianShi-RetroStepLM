"""
Functional ML retrosynthesis implementations.
"""

import logging
import os
import random
from typing import Any, Dict

from rdkit import Chem

from external.desp.inference.retro_predictor import RetroPredictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def format_reactant(smiles, bb_checker):
    """Format a single reactant entry: SMILES plus building-block annotation."""
    return {
        "SMILES": smiles,
        "building_block": bb_checker.check(smiles)[0],
    }


def predict_templates_impl(
    predictor: RetroPredictor, molecule: str, top_k: int = 20
) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(molecule)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {molecule}")
    molecule_canon = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    if top_k is None or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got: {top_k}")
    try:
        predict = getattr(predictor, "predict", None)
        if predict is not None:
            predictions = predict(molecule_canon, top_n=int(top_k))
        else:
            run = getattr(predictor, "run")
            model_top_k = min(int(top_k), int(getattr(predictor, "num_res", top_k)))
            predictions = run(molecule_canon, topk=model_top_k)
        if isinstance(predictions, dict):
            reactants = predictions.get("reactants", [])
            predictions = [
                {"reactants": reactants_smiles.split("."), "template": None}
                for reactants_smiles in reactants
            ]
    except Exception as e:
        raise ValueError(f"ML prediction failed: {e}")
    if not predictions:
        return {
            "SMILES": molecule_canon,
            "predictions": {},
            "internal_data": [],
        }
    # The standard RetroAgent protocol hides candidate rank from the planner.
    # The dedicated DESP MLP Top-10 server opts out so it exposes the native
    # template-ranking order used by the original DESP predictor.
    if os.environ.get("RETRO_MLP_PRESERVE_ORDER") != "1":
        random.shuffle(predictions)
    formatted_predictions = {}
    internal_data = []
    for i, pred in enumerate(predictions):
        template = pred["template"]
        templates_list = template if isinstance(template, list) else [template]
        reactants = pred["reactants"]
        # Store full data internally for expand() lookup by index
        internal_data.append(
            {
                "template": templates_list[0] if templates_list else None,
                "reactants": reactants,
            }
        )
        # Return indexed reactants to agent (save tokens)
        formatted_predictions[i] = reactants
    return {
        "SMILES": molecule_canon,
        "predictions": formatted_predictions,
        "internal_data": internal_data,
    }


def predict_templates(
    predictor: RetroPredictor,
    graph,
    molecule_id: str,
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    Predict one-step retrosynthesis decompositions for a molecule node.

    Args:
        predictor: Initialized RetroPredictor.
        graph: Graph memory instance that stores predictions.
        molecule_id: Molecule node ID to decompose (e.g., "M1").
        top_k: Number of predictions to return.

    Returns:
        Dictionary with indexed predictions (P1, P2, ...). Each prediction
        is a list of {SMILES, building_block} dicts. Pass the prediction ID
        (e.g., "P1") to expand().
        Predictions that would create cycles or duplicate existing reactions
        are automatically filtered out.
        Calling again for the same molecule reuses cached predictions
        (stable IDs, zero budget cost) with updated filtering.
    """
    if molecule_id not in graph.graph:
        raise ValueError(f"Unknown molecule_id: {molecule_id}")
    node = graph.graph.nodes[molecule_id]
    if node.get("type") != "molecule":
        raise ValueError(f"{molecule_id} is not a molecule node")
    molecule = node.get("smiles")

    # If predictions are already cached, skip the ML model call and
    # just re-filter against the current graph state. This keeps
    # prediction IDs (P1, P2, ...) stable and costs zero budget.
    cached = graph.ml_predictions.get(molecule_id)
    if cached is not None:
        bb_checker = graph.bb_checker
        annotated = {}
        n_filtered = 0
        for i, pred in enumerate(cached):
            reactants = pred.get("reactants", [])
            if graph.find_cycle_reactants(molecule_id, reactants):
                n_filtered += 1
                continue
            if graph.would_duplicate_reaction(molecule_id, reactants):
                n_filtered += 1
                continue
            annotated[f"P{i + 1}"] = [format_reactant(r, bb_checker) for r in reactants]
        resp = {"molecule_id": molecule_id, "predictions": annotated}
        if n_filtered > 0:
            resp["filtered"] = f"{n_filtered} predictions removed (cycle or duplicate)"
        if not annotated:
            resp["message"] = (
                "No viable predictions remain for this molecule. "
                "Try a different molecule from the frontier."
            )
            frontier = graph.get_frontier()
            if frontier and all(
                graph.ml_predictions.get(mid) is not None
                and not any(
                    not graph.find_cycle_reactants(mid, p.get("reactants", []))
                    and not graph.would_duplicate_reaction(mid, p.get("reactants", []))
                    for p in graph.ml_predictions.get(mid, [])
                )
                for mid in frontier
            ):
                resp["dead_frontier"] = True
        return resp

    # First call for this molecule in this session: invoke the ML model, shuffle
    # (unless the dedicated DESP Top-10 service preserves rank), filter, and cache.
    # The standard protocol requests at least 50 raw predictions then keeps top_k.
    #
    # Budget/caching semantics (Retro-R1 iterative protocol; intentional):
    #   - Within a session, the filtered cache below makes repeat predict_templates
    #     calls for the same molecule return stable IDs at zero budget cost.
    #   - Across the independent rollouts of an evaluation, each rollout is a fresh
    #     session, so this branch runs again and re-samples a new subset under a new
    #     shuffle. Evaluation charges budget only for molecule SMILES not seen in an
    #     earlier rollout (see slime_retro/eval.py), so a revisited molecule can
    #     surface previously-unshown candidates at zero additional budget cost.
    #   This matches the Retro-R1 evaluation protocol we compare against; it is a
    #   deliberate choice, not a caching bug. To make per-molecule candidates fully
    #   stable across rollouts instead, either seed this shuffle by canonical SMILES
    #   or persist the raw top-50 in a target-scoped (cross-session) cache.
    raw_top_k = top_k if os.environ.get("RETRO_MLP_EXACT_TOP_K") == "1" else max(top_k, 50)
    result = predict_templates_impl(predictor, molecule, top_k=raw_top_k)
    raw_internal = result.pop("internal_data", [])
    raw_predictions = result.get("predictions", {})

    # Pre-filter: remove cycle-causing and duplicate-reaction predictions,
    # then truncate to top_k.
    filtered_internal = []
    filtered_predictions = {}
    n_filtered = 0
    for idx in sorted(raw_predictions.keys()):
        if len(filtered_internal) >= top_k:
            break
        reactants = raw_predictions[idx]
        if graph.find_cycle_reactants(molecule_id, reactants):
            n_filtered += 1
            continue
        if graph.would_duplicate_reaction(molecule_id, reactants):
            n_filtered += 1
            continue
        new_idx = len(filtered_internal)
        filtered_predictions[new_idx] = reactants
        filtered_internal.append(raw_internal[idx])

    graph.register_ml_predictions(molecule_id, filtered_internal)

    bb_checker = graph.bb_checker
    annotated = {}
    for idx, reactants in filtered_predictions.items():
        annotated[f"P{idx + 1}"] = [format_reactant(r, bb_checker) for r in reactants]
    resp = {"molecule_id": molecule_id, "predictions": annotated}
    if n_filtered > 0:
        resp["filtered"] = f"{n_filtered} predictions removed (cycle or duplicate)"
    if not annotated:
        resp["message"] = (
            "No viable predictions remain for this molecule. "
            "Try a different molecule from the frontier."
        )
        frontier = graph.get_frontier()
        if frontier and all(
            graph.ml_predictions.get(mid) is not None
            and not any(
                not graph.find_cycle_reactants(mid, p.get("reactants", []))
                and not graph.would_duplicate_reaction(mid, p.get("reactants", []))
                for p in graph.ml_predictions.get(mid, [])
            )
            for mid in frontier
        ):
            resp["dead_frontier"] = True
    return resp
