"""
AND-OR Tree core data structures for multi-path retrosynthesis planning.
"""

from dataclasses import dataclass, field
from typing import AbstractSet, Any, Optional
from enum import Enum


class NodeStatus(Enum):
    UNSOLVED = "unsolved"
    SOLVED = "solved"
    AVAILABLE = "available"
    FAILED = "failed"
    PENDING = "pending"
    PRUNED = "pruned"


class NodeType(Enum):
    ROOT = "root"
    MOL = "mol"
    REACTION = "reaction"


@dataclass
class TacticEdge:
    """A non-skeletal step (FGI/protection/activation) recorded inside a
    MOL node's instance pool. Links one instance (more stock-reachable)
    to another via a concrete reaction whose abstract product equals
    the abstract reactant — i.e. no new core-core bond formed."""

    src_smiles: str           # the candidate's reactant — the "lower-stock-distance" side
    dst_smiles: str           # the existing instance — the "current" side
    rxn_smiles: str           # full canonical concrete rxn for the activation step
    score: Optional[float] = None
    rationale: str = ""


@dataclass
class MolInstance:
    """One concrete realization of an abstract MOL node. Multiple instances
    share the same `abstract_smiles` of their parent MOL, but carry the
    full concrete SMILES used for stock lookup and single-step expansion."""

    smiles: str               # canonical concrete SMILES
    score: Optional[float] = None
    feasibility: Optional[str] = None
    rationale: str = ""
    source: str = "candidate"  # "root" | "candidate" | "self_loop"
    expanded: bool = False     # vLLM has been called on this instance
    failed: bool = False       # all downstream of this instance is FAILED
    tactic_edges: list[TacticEdge] = field(default_factory=list)
    available: bool = False    # set True when this concrete SMILES hits stock


@dataclass
class ConcreteCandidate:
    """A specific concrete candidate inside a REACTION cluster's pool.
    Shares the parent REACTION's `abstract_rxn_key`, but carries its
    own reactants for tactical materialization."""

    reactants: list           # [[smi, avail], ...] in worker-output shape
    score: Optional[float] = None
    feasibility: Optional[str] = None
    rationale: str = ""
    score_source: str = "single_step"
    rxn_smiles: str = ""
    abstract_rxn_key: str = ""
    is_representative: bool = False
    mapping_failed: bool = False


@dataclass
class TreeNode:
    node_id: str
    node_type: NodeType
    smiles: str
    status: NodeStatus = NodeStatus.UNSOLVED
    children: list["TreeNode"] = field(default_factory=list)
    parent_id: Optional[str] = None
    reaction_candidates: list = field(default_factory=list)
    selected_reaction_indices: list[int] = field(default_factory=list)
    reaction_idx: int = -1
    availability: str = "unavailable"
    depth: int = 0
    product_smiles: Optional[str] = None
    prune_type: Optional[str] = None
    prune_reason: Optional[str] = None
    fail_reason: Optional[str] = None
    strategy_notes: Optional[str] = None
    score: float = 0.0
    visits: int = 0
    prior_score: float = 0.5
    ucb_score: float = 0.0
    last_improved_step: int = 0
    # ---- v2.1 abstraction-clustering additions -----------------------
    # MOL-only: identity / dedup key under the sphere abstraction.
    abstract_smiles: Optional[str] = None
    # MOL-only: pool of equivalent concrete realizations of this node.
    # The "primary" instance (the one currently representing this MOL
    # in active search) is whichever entry has expanded=True; others
    # are queued for instance-level fallback.
    instances: list[MolInstance] = field(default_factory=list)
    # REACTION-only: identity of the abstract cluster this REACTION
    # represents (e.g. "[1*]Ph.[2*]Ph >> Ph-Ph" for any Suzuki variant).
    abstract_rxn_key: Optional[str] = None
    # REACTION-only: same-cluster concrete candidates not chosen as
    # representative — consulted before AND-OR backtrack.
    candidate_pool: list[ConcreteCandidate] = field(default_factory=list)

    def __repr__(self) -> str:
        status_icon = (
            "✓"
            if self.status == NodeStatus.SOLVED
            else ("✓" if self.status == NodeStatus.AVAILABLE else "?")
        )
        type_icon = (
            "ROOT"
            if self.node_type == NodeType.ROOT
            else ("MOL" if self.node_type == NodeType.MOL else "RXN")
        )
        return f"[{self.node_id}] {type_icon}: {self.smiles[:30]}... ({self.status.value}) {status_icon}"


@dataclass
class AOTreeState:
    root: TreeNode
    all_nodes: dict[str, TreeNode] = field(default_factory=dict)
    unsolved_nodes: list[str] = field(default_factory=list)
    solved_paths: list[list[str]] = field(default_factory=list)
    solved_path_count: int = 0
    total_steps: int = 0
    node_counter: int = 0
    starting_mols: AbstractSet[str] = field(default_factory=frozenset)
    target_smiles: str = ""
    _unsolved_set: set = field(default_factory=set, repr=False)
    expanded_smiles: set = field(default_factory=set)
    failed_smiles: set = field(default_factory=set)
    smiles_to_candidates: dict = field(default_factory=dict)
    current_step: int = 0
    # ---- v2.1: abstract MOL identity → node_id -----------------------
    # Replaces the canonical-SMILES dedup channel. Two retro candidates
    # whose reactant MOLs share an abstract SMILES point at the SAME
    # MOL node, with their concrete forms appended as MolInstance
    # entries. Concrete-level dedup still happens inside each node's
    # instances list for stock checks and worker re-dispatch.
    abstract_smiles_to_node_id: dict[str, str] = field(default_factory=dict)
    # Optional metrics bag for end-of-round telemetry (compression
    # ratio, self-loop counts, instance fallbacks). Populated by the
    # state mutators in P2.B/P2.C; read by SSE/CLI in P3.
    abstraction_metrics: dict[str, Any] = field(default_factory=dict)
    # P3.B SSE event queue. State mutators append abstraction-related
    # events here (cluster_formed / self_loop_registered /
    # instance_fallback); the dispatcher drains and forwards them to
    # the progress callback after each expansion. Keeps state-layer
    # self-contained and dispatcher in charge of transport.
    pending_events: list[dict[str, Any]] = field(default_factory=list)

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        return self.all_nodes.get(node_id)

    def is_unsolved(self, node_id: str) -> bool:
        return node_id in self._unsolved_set

    def add_node(self, node: TreeNode) -> None:
        self.all_nodes[node.node_id] = node
        if node.node_type == NodeType.MOL and node.status == NodeStatus.UNSOLVED:
            if node.node_id not in self._unsolved_set:
                self._unsolved_set.add(node.node_id)
                self.unsolved_nodes.append(node.node_id)

    def remove_from_unsolved(self, node_id: str) -> None:
        if node_id in self._unsolved_set:
            self._unsolved_set.discard(node_id)
            self.unsolved_nodes.remove(node_id)
