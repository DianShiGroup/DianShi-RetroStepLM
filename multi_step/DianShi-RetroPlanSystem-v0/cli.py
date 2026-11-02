#!/usr/bin/env python3
"""Standalone, config-driven benchmark CLI; no web backend required."""
import argparse, gc, json, logging, os, sys
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime.config import load_config
from runtime.aot.aot_state import convert_path_to_standard_json
from runtime.retrosynthesis_service import run_retrosynthesis, validate_smiles
from runtime.step_api_client import build_embedded_step_api_client

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

def path(root, value, name):
    if not isinstance(value, str) or not value.strip(): raise RuntimeError(f"config 'run.{name}' must be a non-empty path")
    return value if os.path.isabs(value) else os.path.join(root, value)

def settings(config, root, override=None):
    run = config.get("run")
    if not isinstance(run, dict): raise RuntimeError("config must define a 'run' section for task mode")
    inp, out = path(root, run.get("input"), "input"), path(root, run.get("output"), "output")
    if not os.path.isfile(inp): raise RuntimeError(f"config run.input does not exist: {inp}")
    try: concurrency = int(run.get("concurrency", 8))
    except (TypeError, ValueError): raise RuntimeError("config 'run.concurrency' must be positive") from None
    if concurrency < 1: raise RuntimeError("config 'run.concurrency' must be positive")
    resume = run.get("resume", True)
    if not isinstance(resume, bool): raise RuntimeError("run.resume must be boolean")
    log = override or run.get("log_file")
    return {"input": inp, "output": out, "concurrency": concurrency, "resume": resume, "log_file": path(root, log, "log_file") if log else None}

def add_log_handler(filename):
    if not filename: return
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    handler = logging.FileHandler(filename, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(handler)

def record(idx, smiles, result, config, max_steps, depth, paths):
    stats = result.get("run_stats") or {}; retro = ((config.get("single_step") or {}).get("retro") or {})
    return {"idx": idx, "target_mol": smiles, "success": bool(result.get("success", False)), "solved_paths": result.get("solved_paths", 0), "termination_reason": result.get("termination_reason", "unknown"), "metrics": {"duration_sec": round(stats.get("duration_ms", 0)/1000, 2), "first_path_time_sec": stats.get("first_path_time_sec"), "single_step_calls": stats.get("single_step_calls", 0), "llm_calls": stats.get("llm_calls", 0), "llm_prompt_tokens": stats.get("llm_prompt_tokens", 0), "llm_completion_tokens": stats.get("llm_completion_tokens", 0)}, "config": {"planner": (config.get("planner") or {}).get("default_model", "unknown"), "single_step_backend": retro.get("backend", "dianshi"), "rag_enabled": bool((config.get("single_step") or {}).get("rag_enabled", True)), "max_steps": max_steps, "max_tree_depth": depth, "target_complete_paths": paths}, "trees": [convert_path_to_standard_json(x.get("steps", [])) for x in result.get("solved_paths_detail", [])]}

def run_case(idx, smiles, config, max_steps, depth, paths, client):
    valid, msg = validate_smiles(smiles)
    if not valid:
        return record(idx, smiles, {"termination_reason": f"Invalid SMILES: {msg}"}, config, max_steps, depth, paths)
    try:
        result = run_retrosynthesis(config=config, smiles=smiles, max_steps=max_steps, max_tree_depth=depth, target_complete_paths=paths, step_api_client=client)
        return record(idx, smiles, result, config, max_steps, depth, paths)
    except Exception as exc:
        logger.exception("Error processing case=%d", idx + 1)
        return record(idx, smiles, {"termination_reason": str(exc)}, config, max_steps, depth, paths)

def read_checkpoint(filename):
    done, old = set(), []
    if not os.path.exists(filename): return done, old
    with open(filename, encoding="utf-8") as f:
        for line in f:
            try: obj = json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(obj.get("idx"), int) and isinstance(obj.get("target_mol"), str): done.add((obj["idx"], obj["target_mol"])); old.append(obj)
    return done, old

def batch(inp, out, config, max_steps, depth, paths, client, concurrency, resume):
    with open(inp, encoding="utf-8") as f: items = [(i, x.strip()) for i, x in enumerate(f) if x.strip()]
    done, records = read_checkpoint(out) if resume else (set(), [])
    pending = [x for x in items if x not in done]; os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "a" if resume and os.path.exists(out) else "w", encoding="utf-8") as f:
        active, pos = {}, 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while pos < len(pending) and len(active) < concurrency:
                i, s = pending[pos]; pos += 1; active[pool.submit(run_case, i, s, config, max_steps, depth, paths, client)] = (i, s)
            while active:
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    i, s = active.pop(future); item = future.result(); f.write(json.dumps(item, ensure_ascii=False)+"\n"); f.flush(); records.append(item); logger.info("[%d/%d completed] case=%d status=%s", len(records), len(items), i+1, "success" if item.get("success") else "failed")
                while pos < len(pending) and len(active) < concurrency:
                    i, s = pending[pos]; pos += 1; active[pool.submit(run_case, i, s, config, max_steps, depth, paths, client)] = (i, s)
                gc.collect()
    n, ok = len(records), sum(1 for x in records if x.get("success")); logger.info("Batch Summary: success_rate=%d/%d (%.1f%%)", ok, n, ok/n*100 if n else 0); logger.info("Done. Results written to %s", out)

def main():
    p = argparse.ArgumentParser(description="Standalone retro synthesis CLI"); p.add_argument("--config"); p.add_argument("--log-file"); p.add_argument("--smiles"); p.add_argument("--in_file"); p.add_argument("--output", default="result.jsonl"); a = p.parse_args()
    if a.smiles and a.in_file: p.error("cannot use both --smiles and --in_file")
    root = os.path.dirname(os.path.abspath(__file__)); cfg_path = a.config or os.path.join(root, "config.yaml"); cfg_path = cfg_path if os.path.isabs(cfg_path) else os.path.join(root, cfg_path)
    if not os.path.isfile(cfg_path): p.error(f"config file not found: {cfg_path}")
    config = load_config(cfg_path); task = not a.smiles and not a.in_file
    if task:
        try: run = settings(config, root, a.log_file)
        except RuntimeError as exc: p.error(str(exc))
    else: run = {"input": a.in_file, "output": a.output, "concurrency": 1, "resume": True, "log_file": a.log_file}
    if not run["resume"] and os.path.exists(run["output"]): p.error("run.resume=false refuses to overwrite existing output")
    add_log_handler(run["log_file"]); client = build_embedded_step_api_client(config); search = config.get("search") or {}; ms, depth, paths = int(search.get("max_steps", 30)), int(search.get("max_tree_depth", 10)), int(search.get("target_complete_paths", 10))
    if a.smiles: run_case(0, a.smiles, config, ms, depth, paths, client)
    else: batch(run["input"], run["output"], config, ms, depth, paths, client, run["concurrency"], run["resume"])

if __name__ == "__main__": main()
