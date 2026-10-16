"""
HTTP server for retrosynthesis tools with per-session state management.

Each rollout episode creates a session (with its own GraphMemory) and invokes
tools via HTTP.  Heavy resources (building-block table, ML models) are loaded once
per process and shared across sessions in that process.

Per-call logging:
    Each ``/invoke_tool`` call
    captures the graph state *before* the tool executes and appends a log entry
    to the session's ``tool_calls_log``.  ``/delete_session`` returns the full
    log alongside the final graph state and solved route steps.

Session affinity:
    Multiple server instances run on consecutive ports (8100, 8101, ...).
    The client assigns new sessions to instances in round-robin order for even
    load distribution, then pins all subsequent requests for that session to
    the same instance.

    Launch N instances with:
        python -m tools.tool_server base_port=8100 num_instances=8

Usage:
    # Single instance:
    python -m tools.tool_server base_port=8100

    # 8 instances on ports 8100-8107:
    python -m tools.tool_server base_port=8100 num_instances=8
"""

import asyncio
import logging
import multiprocessing
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache, partial

import chz
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from data.building_blocks import BuildingBlockChecker
from data.loaders import (
    RETRO_MODEL_PATH,
    RETRO_TEMPLATES_PATH,
)
from external.desp.inference.retro_predictor import RetroPredictor
from inference.support.config_loader import load_yaml
from inference.step.dianshi_direct import load as load_dianshi_direct
from slime_retro.config import EXPAND_MODE
from tools.heuristics import (
    calculate_synthetic_accessibility,
    calculate_scscore,
)
from tools.memory import (
    MAX_ROUTE_DEPTH,
    GraphMemory,
    expand_single,
    expand_multi,
    get_frontier,
    inspect_node,
)
from tools.ml_retro import predict_templates
from tools.scscore import SCScorer
from tools.search import (
    search_building_blocks,
    get_molecule_info,
)
from tools.tools import from_callable

logger = logging.getLogger(__name__)


@dataclass
class ToolStorage:
    bb_checker: BuildingBlockChecker
    retro_predictor: RetroPredictor
    scscore_model: SCScorer


@lru_cache(maxsize=1)
def load_tool_storage():
    bb_checker = BuildingBlockChecker()
    adapter_config = os.environ.get("RETRO_SINGLE_STEP_CONFIG")
    if adapter_config:
        cfg = load_yaml(adapter_config) or {}
        backend = cfg.get("backend") or (cfg.get("retro") or {}).get("backend")
        backend = backend or ("dianshi_direct" if (cfg.get("retro") or {}).get("api_url") else None)
        loaders = {
            "dianshi_direct": load_dianshi_direct,
        }
        if backend not in loaders:
            raise ValueError(f"Unsupported single-step backend: {backend}")
        retro_predictor = loaders[backend](adapter_config)
    else:
        retro_model_path = os.environ.get("RETRO_MLP_MODEL_PATH", RETRO_MODEL_PATH)
        retro_templates_path = os.environ.get(
            "RETRO_MLP_TEMPLATES_PATH", RETRO_TEMPLATES_PATH
        )
        retro_predictor = RetroPredictor(retro_model_path, retro_templates_path)
    logger.info("Loaded single-step predictor (%s)", type(retro_predictor).__name__)
    scscore_model = SCScorer()
    logger.info(f"Loaded SCScore ({len(scscore_model.vars)} layers)")
    return ToolStorage(bb_checker, retro_predictor, scscore_model)


def build_tool_map(graph, tool_storage, predict_templates_top_k):
    tools = {
        "get_molecule_info": from_callable(get_molecule_info, name="get_molecule_info"),
        "calculate_synthetic_accessibility": from_callable(
            calculate_synthetic_accessibility
        ),
        "search_building_blocks": from_callable(
            partial(search_building_blocks, tool_storage.bb_checker),
            name="search_building_blocks",
        ),
        "predict_templates": from_callable(
            partial(
                predict_templates,
                tool_storage.retro_predictor,
                graph,
                top_k=predict_templates_top_k,
            ),
            name="predict_templates",
        ),
        "calculate_scscore": from_callable(
            partial(calculate_scscore, tool_storage.scscore_model),
            name="calculate_scscore",
        ),
        "expand": from_callable(
            partial(expand_multi if EXPAND_MODE == "multi" else expand_single, graph),
            name="expand",
        ),
        "inspect_node": from_callable(
            partial(inspect_node, graph), name="inspect_node"
        ),
        "get_frontier": from_callable(
            partial(get_frontier, graph), name="get_frontier"
        ),
    }
    return tools


# Global shared state (per-process; each instance has its own)
tool_storage = None
sessions = {}

# Precomputed tool schemas in OpenAI format (built once during lifespan).
tool_schemas = []

# Thread pool for concurrent invoke_tool calls per instance.
# With client semaphore=128 and 16 instances, max ~8 concurrent per instance.
# Each thread uses OMP_NUM_THREADS (24) internally, so total = 8 * 24 = 192.
THREAD_POOL_SIZE = int(os.environ.get("RETRO_TOOL_THREADS_PER_INSTANCE", 8))
MAX_ROUTE_DEPTH_CFG = int(os.environ.get("RETRO_MAX_ROUTE_DEPTH", MAX_ROUTE_DEPTH))
FIXED_PREDICT_TEMPLATES_TOP_K = os.environ.get("RETRO_FIXED_PREDICT_TEMPLATES_TOP_K")


@asynccontextmanager
async def lifespan(app):
    global tool_storage, tool_schemas
    logger.info(
        "Loading tool storage (building blocks, ML models, SCScore)... pid=%d",
        os.getpid(),
    )
    tool_storage = load_tool_storage()
    # Set the default executor for asyncio.to_thread()
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE))

    # Build tool schemas once (schemas are target-independent)
    dummy_graph = GraphMemory(
        target="C",
        bb_checker=tool_storage.bb_checker,
        max_route_depth=MAX_ROUTE_DEPTH_CFG,
    )
    dummy_tools = build_tool_map(dummy_graph, tool_storage, 20)
    tool_schemas = [t.to_schema() for t in dummy_tools.values()]

    logger.info(
        "Tool storage loaded, thread pool=%d, %d tool schemas (pid=%d)",
        THREAD_POOL_SIZE,
        len(tool_schemas),
        os.getpid(),
    )
    yield
    sessions.clear()
    close_predictor = getattr(tool_storage.retro_predictor, "close", None)
    if close_predictor is not None:
        close_predictor()
    logger.info("Tool server shut down, sessions cleared (pid=%d)", os.getpid())


app = FastAPI(lifespan=lifespan)


class CreateSessionRequest(BaseModel):
    target: str
    predict_templates_top_k: int = 20
    session_id: str  # assigned by client for hash-based routing


class InvokeToolRequest(BaseModel):
    session_id: str
    tool_name: str
    arguments: dict


class DeleteSessionRequest(BaseModel):
    session_id: str


class GetSessionLogRequest(BaseModel):
    session_id: str


@app.post("/create_session")
async def create_session(req: CreateSessionRequest):
    """Create a new tool session with its own GraphMemory for a target molecule."""
    t0 = time.time()
    session_id = req.session_id
    predict_templates_top_k = (
        int(FIXED_PREDICT_TEMPLATES_TOP_K)
        if FIXED_PREDICT_TEMPLATES_TOP_K is not None
        else req.predict_templates_top_k
    )
    graph = GraphMemory(
        target=req.target,
        bb_checker=tool_storage.bb_checker,
        max_route_depth=MAX_ROUTE_DEPTH_CFG,
    )
    tools = build_tool_map(graph, tool_storage, predict_templates_top_k)
    sessions[session_id] = {
        "graph": graph,
        "tools": tools,
        "target": req.target,
        "created_at": t0,
        "tool_calls_log": [],
        "call_counter": 0,
    }
    elapsed = time.time() - t0
    logger.info(
        f"create_session {session_id} ({elapsed:.3f}s) target={req.target[:50]} sessions={len(sessions)}"
    )
    return {"session_id": session_id}


@app.post("/invoke_tool")
async def invoke_tool(req: InvokeToolRequest):
    """Invoke a tool within a session. Returns the tool result dict.

    Invokes a tool within a session and appends a log entry with the
    tool name, arguments, result, and call ID.
    """
    session = sessions.get(req.session_id)
    if session is None:
        return {"error": f"Unknown session: {req.session_id}"}
    tool = session["tools"].get(req.tool_name)
    if tool is None:
        return {"error": f"Unknown tool: {req.tool_name}"}

    t0 = time.time()
    result = await asyncio.to_thread(tool.invoke, req.arguments)
    elapsed = time.time() - t0

    session["call_counter"] += 1
    session["tool_calls_log"].append(
        {
            "name": req.tool_name,
            "args": req.arguments,
            "result": result,
            "call_id": session["call_counter"],
        }
    )

    # Auto-check if root is SOLVED after every tool call (O(1) lookup).
    graph = session["graph"]
    if graph.graph.nodes[graph.root_id].get("status") == "SOLVED":
        result["route_solved"] = True

    logger.info(
        f"invoke_tool {req.session_id} {req.tool_name} ({elapsed:.3f}s) sessions={len(sessions)}"
    )
    return result


@app.post("/delete_session")
async def delete_session(req: DeleteSessionRequest):
    """Delete a tool session, returning graph state, tool log, and route steps."""
    session = sessions.pop(req.session_id, None)
    if session is None:
        return {"error": f"Unknown session: {req.session_id}"}
    graph = session["graph"]
    try:
        route_tree = graph.get_solved_route_tree()
    except ValueError:
        route_tree = None  # root not solved
    return {
        "graph_state": graph.to_dict(),
        "tool_calls_log": session["tool_calls_log"],
        "route_steps": graph.get_solved_route_linear(),
        "route_tree": route_tree,
    }


@app.post("/get_session_log")
async def get_session_log(req: GetSessionLogRequest):
    """Read tool_calls_log without deleting the session."""
    session = sessions.get(req.session_id)
    if session is None:
        return {"error": f"Unknown session: {req.session_id}"}
    return {
        "tool_calls_log": session["tool_calls_log"],
    }


@app.get("/metrics")
async def metrics():
    """Return server metrics for monitoring."""
    now = time.time()
    session_list = []
    for sid, s in sessions.items():
        session_list.append(
            {
                "session_id": sid,
                "target": s["target"],
                "calls": s["call_counter"],
                "age_s": round(now - s["created_at"], 1),
            }
        )
    return {
        "pid": os.getpid(),
        "active_sessions": len(sessions),
        "total_calls": sum(s["call_counter"] for s in sessions.values()),
        "thread_pool_size": THREAD_POOL_SIZE,
        "sessions": session_list,
    }


@app.get("/tool_schemas")
async def get_tool_schemas():
    """Return precomputed OpenAI-style tool schemas."""
    return {"schemas": tool_schemas}


def run_instance(host, port):
    """Run a single uvicorn instance (called in a subprocess)."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [port={port}] %(message)s",
    )
    uvicorn.run(
        "tools.tool_server:app",
        host=host,
        port=port,
        workers=1,
    )


@chz.chz
class ToolServerConfig:
    host: str = "0.0.0.0"
    """Server bind host."""

    base_port: int = 8100
    """Base port. Instances use base_port, base_port+1, ..."""

    num_instances: int = 1
    """Number of server instances (separate processes on consecutive ports).
    Each loads its own ToolStorage. Client assigns sessions round-robin."""


def main():
    args = chz.entrypoint(ToolServerConfig)

    # Set thread limits before spawning subprocesses so child processes inherit them.
    n_threads = str(os.environ.get("RETRO_TOOL_THREADS_PER_PROCESS", 24))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, n_threads)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    logger.info(
        "Starting %d tool server instance(s) on ports %d-%d",
        args.num_instances,
        args.base_port,
        args.base_port + args.num_instances - 1,
    )
    for i in range(args.num_instances):
        port = args.base_port + i
        p = multiprocessing.Process(
            target=run_instance,
            args=(args.host, port),
            name=f"tool-server-{port}",
            daemon=True,
        )
        p.start()
        logger.info("  Instance %d: port %d (pid=%d)", i, port, p.pid)

    # Block until SIGINT/SIGTERM; daemon processes are cleaned up on exit
    signal.pause()


if __name__ == "__main__":
    main()
