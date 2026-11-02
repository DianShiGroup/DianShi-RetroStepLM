"""
Core AND-OR graph memory for retrosynthesis planning.

OR-nodes are molecules (M#). AND-nodes are reactions (R#).
"""

import random
from typing import Dict, Any, List

import networkx as nx

from utils.smiles import canonicalize_smiles

MAX_ROUTE_DEPTH = 30


class GraphMemory:
    """AND-OR graph with molecule (OR) and reaction (AND) nodes (M#/R#)."""

    def __init__(self, target, bb_checker, max_route_depth=MAX_ROUTE_DEPTH):
        self.graph = nx.DiGraph()
        self.bb_checker = bb_checker
        self.max_route_depth = max_route_depth
        self.ml_predictions = {}
        self.reaction_index = {}
        self.smiles_to_mol_id = {}
        self.next_molecule_id = 1
        self.next_reaction_id = 1

        target = canonicalize_smiles(target)
        self.root_id = self.get_or_create_molecule(target)

    @staticmethod
    def reaction_key(product, reactants):
        reactants_key = ".".join(sorted(reactants))
        return f"{product}|{reactants_key}"

    def new_molecule_id(self):
        mid = f"M{self.next_molecule_id}"
        self.next_molecule_id += 1
        return mid

    def new_reaction_id(self):
        rid = f"R{self.next_reaction_id}"
        self.next_reaction_id += 1
        return rid

    def propagate_depth(self, mol_id, new_depth):
        """BFS propagation: update descendant molecule depths if shorter path found.

        Only updates a descendant if the new depth (via this ancestor) is
        actually shorter than its current depth. Matches Retro-R1's
        level-by-level step_dict propagation.
        """
        self.graph.nodes[mol_id]["depth"] = new_depth
        # BFS through reaction→molecule edges, depth increments at each reaction
        current_level = [mol_id]
        child_depth = new_depth + 1
        while current_level:
            next_level = []
            for mid in current_level:
                # molecule → reaction edges
                for rid in self.graph.successors(mid):
                    if self.graph.nodes[rid].get("type") != "reaction":
                        continue
                    # reaction → molecule edges
                    for child_mid in self.graph.successors(rid):
                        child_node = self.graph.nodes[child_mid]
                        if child_node.get(
                            "type"
                        ) == "molecule" and child_depth < child_node.get(
                            "depth", float("inf")
                        ):
                            child_node["depth"] = child_depth
                            next_level.append(child_mid)
            current_level = next_level
            child_depth += 1

    def get_or_create_molecule(self, smiles, depth=0):
        smiles = canonicalize_smiles(smiles)
        existing = self.smiles_to_mol_id.get(smiles)
        if existing:
            node = self.graph.nodes[existing]
            if depth < node.get("depth", float("inf")):
                self.propagate_depth(existing, depth)
            return existing
        mid = self.new_molecule_id()
        ok, _ = self.bb_checker.check(smiles)
        if ok:
            status = "SOLVED"
        else:
            status = "OPEN"
        self.graph.add_node(
            mid,
            type="molecule",
            id=mid,
            smiles=smiles,
            status=status,
            bb_status="building_block" if ok else "not_building_block",
            depth=depth,
        )
        self.smiles_to_mol_id[smiles] = mid
        return mid

    def register_ml_predictions(self, molecule_id, predictions):
        """Register ML predictions for a molecule node (first call only).

        Subsequent calls for the same molecule are no-ops to keep
        prediction IDs (P1, P2, ...) stable across the episode.
        """
        if molecule_id not in self.ml_predictions:
            self.ml_predictions[molecule_id] = list(predictions or [])

    def get_prediction_by_id(self, molecule_id, prediction_id):
        """Look up prediction data by molecule ID and prediction ID (e.g., 'P1')."""
        predictions = self.ml_predictions.get(molecule_id, [])
        if not predictions:
            raise ValueError(
                f"No ML predictions found for {molecule_id}. "
                "Call predict_templates first."
            )
        if not isinstance(prediction_id, str) or not prediction_id.startswith("P"):
            raise ValueError(
                f"prediction_id must be a prediction ID like 'P1', got: {prediction_id}"
            )
        try:
            idx = int(prediction_id[1:])
        except ValueError:
            raise ValueError(
                f"Invalid prediction_id: {prediction_id}. Expected format: P1, P2, ..."
            )
        if idx < 1 or idx > len(predictions):
            raise ValueError(
                f"{prediction_id} out of range. "
                f"{molecule_id} has {len(predictions)} predictions (P1–P{len(predictions)})."
            )
        return predictions[idx - 1]

    def evaluate_node_status(self, node_id):
        node = self.graph.nodes[node_id]
        children = list(self.graph.successors(node_id))
        if node.get("type") == "reaction":
            if children and all(
                self.graph.nodes[c].get("status") == "SOLVED" for c in children
            ):
                return "SOLVED"
        elif node.get("type") == "molecule":
            if node.get("bb_status") == "building_block":
                return "SOLVED"
            if children and any(
                self.graph.nodes[c].get("status") == "SOLVED" for c in children
            ):
                return "SOLVED"
        return "OPEN"

    def propagate_status(self, start_node_id):
        current_ids = {start_node_id}
        while current_ids:
            next_parents = set()
            for node_id in current_ids:
                for parent_id in self.graph.predecessors(node_id):
                    parent = self.graph.nodes[parent_id]
                    old_status = parent.get("status")
                    new_status = self.evaluate_node_status(parent_id)
                    if new_status != old_status:
                        parent["status"] = new_status
                        next_parents.add(parent_id)
            current_ids = next_parents

    def find_cycle_reactants(self, molecule_id, reactants):
        """Return list of reactant node IDs that would create a directed cycle."""
        cycle_ids = []
        for r in reactants:
            r_canon = canonicalize_smiles(r) or r
            existing_id = self.smiles_to_mol_id.get(r_canon)
            if not existing_id:
                continue
            if existing_id == molecule_id or nx.has_path(
                self.graph, existing_id, molecule_id
            ):
                cycle_ids.append(existing_id)
        return cycle_ids

    def would_duplicate_reaction(self, molecule_id, reactants):
        """Check if this exact reaction already exists in the graph."""
        product = self.graph.nodes[molecule_id].get("smiles")
        reactants_canon = [canonicalize_smiles(r) or r for r in reactants]
        key = self.reaction_key(product, reactants_canon)
        return key in self.reaction_index

    def expand(self, molecule_id, prediction_id, rationale=None):
        if molecule_id not in self.graph:
            raise ValueError("Unknown molecule_id")
        mol_node = self.graph.nodes[molecule_id]
        if mol_node.get("type") != "molecule":
            raise ValueError("molecule_id must refer to a molecule node")
        if mol_node.get("status") == "SOLVED":
            raise ValueError(
                f"{molecule_id} is already SOLVED. No need to expand further."
            )

        depth = mol_node.get("depth", 0)
        if depth >= self.max_route_depth:
            raise ValueError(
                f"Route depth limit reached: {molecule_id} at depth {depth} (max {self.max_route_depth})"
            )
        product = mol_node.get("smiles")

        # Look up full prediction data by ID
        pred = self.get_prediction_by_id(molecule_id, prediction_id)
        reactants = pred.get("reactants", [])
        reactants = [canonicalize_smiles(r) or r for r in reactants]

        key = self.reaction_key(product, reactants)
        if key in self.reaction_index:
            raise ValueError(f"Reaction already exists: id={self.reaction_index[key]}")

        # Revalidate cycles at expansion time. Cached predictions are cycle-filtered
        # when retrieved, but the graph may have changed since (another branch could
        # have added a path making a reactant an ancestor of this molecule). Without
        # this check, expand() could introduce a directed cycle into the AND-OR graph.
        cycle_ids = self.find_cycle_reactants(molecule_id, reactants)
        if cycle_ids:
            raise ValueError(
                f"Expansion would create a cycle: reactant(s) {cycle_ids} "
                f"are ancestors of {molecule_id}"
            )

        rid = self.new_reaction_id()
        self.reaction_index[key] = rid

        # Build reactant nodes first (reactants are one reaction deeper)
        new_molecules = []
        reactant_ids = []
        for r in reactants:
            mid = self.get_or_create_molecule(r, depth=depth + 1)
            reactant_ids.append(mid)
            child = self.graph.nodes[mid]
            new_molecules.append(
                {
                    "id": mid,
                    "SMILES": child.get("smiles"),
                    "status": child.get("status"),
                }
            )

        self.graph.add_node(
            rid,
            type="reaction",
            id=rid,
            status="OPEN",
            rationale=rationale,
            product_id=molecule_id,
            product_smiles=product,
            reactants=reactants,
            reactant_ids=reactant_ids,
        )
        self.graph.add_edge(molecule_id, rid)
        for mid in reactant_ids:
            self.graph.add_edge(rid, mid)
        self.graph.nodes[rid]["status"] = self.evaluate_node_status(rid)
        self.propagate_status(rid)

        return {
            "prediction_id": prediction_id,
            "reaction_id": rid,
            "new_molecules": new_molecules,
            "reaction_status": self.graph.nodes[rid]["status"],
        }

    def get_frontier(self):
        """Return OPEN leaf molecules (unexpanded, no child reactions).

        Items are shuffled to prevent positional bias during RL training.
        Returns empty dict when all molecules are solved.
        """
        if self.root_id in self.graph:
            reachable = {self.root_id} | nx.descendants(self.graph, self.root_id)
        else:
            reachable = set()

        frontier = []
        for node_id, data in self.graph.nodes(data=True):
            if node_id not in reachable:
                continue
            if data.get("type") != "molecule":
                continue
            if data.get("status") != "OPEN":
                continue
            if self.graph.out_degree(node_id) == 0:
                frontier.append((node_id, data.get("smiles")))

        random.shuffle(frontier)
        return dict(frontier)

    def build_tree(
        self,
        root_id,
        status_filter=None,
        reachable=None,
        include_reactants=False,
        sort_children=False,
        shuffle_children=False,
    ):
        def iter_children(node_id):
            children = list(self.graph.successors(node_id))
            if sort_children:
                return sorted(
                    children, key=lambda n: int(n[1:]) if n[1:].isdigit() else n
                )
            if shuffle_children:
                random.shuffle(children)
            return children

        def should_include(node_id):
            if reachable is not None and node_id not in reachable:
                return False
            if status_filter:
                node = self.graph.nodes[node_id]
                if node.get("status") != status_filter:
                    return False
            return True

        if root_id not in self.graph or not should_include(root_id):
            return None

        result = None
        stack = [
            {
                "node_id": root_id,
                "entered": False,
                "iter": None,
                "children": [],
                "path": set(),
            }
        ]

        while stack:
            frame = stack[-1]
            node_id = frame["node_id"]

            if not frame["entered"]:
                frame["entered"] = True
                node = self.graph.nodes[node_id]
                if node_id in frame["path"]:
                    cycle_node = {
                        "id": node_id,
                        "type": node.get("type"),
                        "status": node.get("status"),
                        "cycle": True,
                        "children": [],
                    }
                    stack.pop()
                    if stack:
                        stack[-1]["children"].append(cycle_node)
                    else:
                        result = cycle_node
                    continue
                frame["node"] = node
                frame["iter"] = iter(iter_children(node_id))
                frame["path"] = frame["path"] | {node_id}

            try:
                child_id = next(frame["iter"])
            except StopIteration:
                node = frame["node"]
                status = node.get("status")
                if node.get("type") == "molecule":
                    node_dict = {
                        "id": node_id,
                        "SMILES": node.get("smiles"),
                        "status": status,
                    }
                else:
                    node_dict = {
                        "id": node_id,
                        "status": status,
                    }
                    if include_reactants:
                        node_dict["reactants"] = node.get("reactants", [])
                if frame["children"]:
                    node_dict["children"] = frame["children"]
                stack.pop()
                if stack:
                    stack[-1]["children"].append(node_dict)
                else:
                    result = node_dict
                continue

            if child_id not in self.graph or not should_include(child_id):
                continue
            stack.append(
                {
                    "node_id": child_id,
                    "entered": False,
                    "iter": None,
                    "children": [],
                    "path": frame["path"],
                }
            )

        return result

    def inspect_node(self, node_id):
        if node_id not in self.graph:
            raise ValueError("Unknown node_id")
        node = self.graph.nodes[node_id]

        # Build parents dict {id: smiles}
        parents = {}
        for pid in self.graph.predecessors(node_id):
            p = self.graph.nodes[pid]
            parents[pid] = p.get("smiles") or p.get("product_smiles")

        # Build children dict {id: smiles}
        children = {}
        for cid in self.graph.successors(node_id):
            c = self.graph.nodes[cid]
            children[cid] = c.get("smiles") or c.get("product_smiles")

        data = {
            "id": node_id,
            "status": node.get("status"),
            "parents": parents,
            "children": children,
        }
        # Add SMILES and building block status for molecule nodes
        if node.get("type") == "molecule":
            data["SMILES"] = node.get("smiles")
            data["building_block"] = node.get("bb_status") == "building_block"
        return data

    def get_solved_route_tree(self):
        root_status = self.graph.nodes[self.root_id].get("status")
        if root_status != "SOLVED":
            raise ValueError("Root is not SOLVED")
        reachable = {self.root_id} | nx.descendants(self.graph, self.root_id)
        return self.build_tree(
            self.root_id,
            status_filter="SOLVED",
            reachable=reachable,
            include_reactants=True,
            sort_children=True,
        )

    def get_solved_route_linear(self):
        try:
            tree = self.get_solved_route_tree()
        except ValueError:
            return []

        steps = []
        root_smiles = tree.get("SMILES")
        if not root_smiles:
            return steps

        stack = [
            {
                "node": tree,
                "state": "enter",
                "current": [root_smiles],
            }
        ]

        while stack:
            frame = stack[-1]
            node = frame["node"]
            if frame["state"] == "enter":
                children = node.get("children") or []
                if not children:
                    current = frame["current"]
                    stack.pop()
                    if stack:
                        stack[-1]["current"] = current
                    continue
                reaction = children[0]
                product = node.get("SMILES")
                reactants = reaction.get("reactants") or []
                updated = [m for m in frame["current"] if m != product] + list(
                    reactants
                )
                step = {
                    "Molecule set": sorted(frame["current"]),
                    "Product": product,
                    "Reactants": reactants,
                    "Updated molecule set": sorted(updated),
                }
                steps.append(step)
                frame["state"] = "children"
                frame["current"] = updated
                frame["children"] = list(reaction.get("children") or [])
                frame["child_idx"] = 0
                continue

            if frame["child_idx"] >= len(frame["children"]):
                current = frame["current"]
                stack.pop()
                if stack:
                    stack[-1]["current"] = current
                continue

            child = frame["children"][frame["child_idx"]]
            frame["child_idx"] += 1
            stack.append({"node": child, "state": "enter", "current": frame["current"]})
        return steps

    def to_dict(self):
        nodes = []
        for n, data in self.graph.nodes(data=True):
            entry = {"id": n, **data}
            nodes.append(entry)
        edges = []
        for u, v in self.graph.edges():
            edges.append({"from": u, "to": v})
        return {
            "format": "bipartite",
            "nodes": nodes,
            "edges": edges,
            "root_id": self.root_id,
        }


def expand_single(
    graph: GraphMemory,
    molecule_id: str,
    prediction_id: str,
) -> Dict[str, Any]:
    """
    Expand a molecule node (M#) using a prediction from predict_templates.

    Args:
        graph: Graph memory instance.
        molecule_id: Molecule node ID to expand (e.g., "M1").
        prediction_id: Prediction ID from predict_templates (e.g., "P1").

    Returns:
        Dictionary with the new reaction and parent status. The agent inspects
        the remaining OPEN molecules with a separate get_frontier() call.
    """
    result = graph.expand(molecule_id, prediction_id)
    result["parent_id"] = molecule_id
    result["parent_status"] = graph.graph.nodes[molecule_id].get("status")
    return result


def expand_multi(
    graph: GraphMemory,
    molecule_id: str,
    prediction_id: str,
) -> Dict[str, Any]:
    """
    Expand a molecule node (M#) using one or more predictions from predict_templates.
    Multiple prediction IDs can be comma-separated (e.g., "P1,P3,P5") to create
    multiple reaction branches at once. Each branch is independent (OR semantics).

    Args:
        graph: Graph memory instance.
        molecule_id: Molecule node ID to expand (e.g., "M1").
        prediction_id: One or more prediction IDs, comma-separated (e.g., "P1" or "P1,P3,P5").

    Returns:
        Dictionary with expansion results and parent status. The agent inspects
        the remaining OPEN molecules with a separate get_frontier() call.
    """
    ids = [p.strip() for p in prediction_id.split(",") if p.strip()]

    expansions = []
    errors = []
    for pid in ids:
        try:
            expansions.append(graph.expand(molecule_id, pid))
        except ValueError as e:
            errors.append({"prediction_id": pid, "error": str(e)})

    response = {
        "expansions": expansions,
        "parent_id": molecule_id,
        "parent_status": graph.graph.nodes[molecule_id].get("status"),
    }
    if errors:
        response["errors"] = errors
    return response


def inspect_node(graph: GraphMemory, node_id: str) -> Dict[str, Any]:
    """
    Return full structural details for a node (M# or R#).

    Args:
        graph: Graph memory instance.
        node_id: Node ID to inspect.
    """
    return graph.inspect_node(node_id)


def get_frontier(graph: GraphMemory) -> Dict[str, Any]:
    """
    Return OPEN leaf molecules that have not been expanded yet.

    Args:
        graph: Graph memory instance.
    """
    return graph.get_frontier()
