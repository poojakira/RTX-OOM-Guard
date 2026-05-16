"""
benchmarks/model_fragmentation.py
=================================
Advanced fragmentation and OOM modeling for RTX-class GPUs.

Tracks:
- OOM rate (baseline vs defrag)
- Runs to first OOM
- Max stable batch size (Largest B that runs K times without OOM)
- GPU utilization and overhead (% time spent in defrag)
- Fragmentation metrics (free blocks, largest block, frag index)
"""

import os
import sys
import json
import argparse
import statistics
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import mlflow

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtx_oom_guard.trainer.callback import DefragCallback
from rtx_oom_guard.utils import get_logger, parse_memory_snapshot, DefragConfig

log = get_logger("model_fragmentation")

# ── Model ────────────────────────────────────────────────────────────────────

class WorkloadModel(nn.Module):
    """GPT-2-style model for memory-intensive workloads."""
    def __init__(self, vocab: int = 50257, d: int = 768, layers: int = 6, heads: int = 12):
        super().__init__()
        self.tok  = nn.Embedding(vocab, d)
        self.pos  = nn.Embedding(1024, d)
        self.enc  = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, d*4, dropout=0.0,
                                       batch_first=True, norm_first=True),
            num_layers=layers,
        )
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        return self.head(self.enc(h))

# ── Fragmentation Engine ──────────────────────────────────────────────────────

def fragment_memory(n_chunks: int, chunk_mb: int = 8) -> list:
    """Create non-contiguous holes in the CUDA pool."""
    pairs = []
    for _ in range(n_chunks):
        big   = torch.empty(chunk_mb * 1024 * 256, device="cuda")   # chunk_mb MB
        small = torch.empty(64 * 1024, device="cuda")               # 256 KB anchor
        pairs.append((big, small))

    survivors = []
    for big, small in pairs:
        del big
        survivors.append(small)
    return survivors

# ── Single Trial ─────────────────────────────────────────────────────────────

def run_trial(use_defrag: bool, cfg: dict) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_mem = torch.cuda.get_device_properties(0).total_memory
    total_mb = total_mem / 1024**2
    
    model = WorkloadModel(layers=cfg["layers"]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    cb = None
    if use_defrag:
        cb = DefragCallback(threshold=0.60)
        cb.on_train_begin()

    util_samples: List[float] = []
    frag_metrics: List[Dict] = []
    step_times: List[float] = []
    cb_overhead_s = 0.0
    oom_hit = False
    oom_step = -1
    anchors = None

    try:
        for step in range(cfg["steps"]):
            t_step_start = time.perf_counter()
            
            # Defrag Overhead tracking (Begin)
            if cb:
                t0 = time.perf_counter()
                cb.on_step_begin()
                cb_overhead_s += (time.perf_counter() - t0)

            try:
                # 1. Fragment memory
                anchors = fragment_memory(cfg["frag_chunks"], cfg["chunk_mb"])

                # 2. Record fragmentation metrics
                snap = parse_memory_snapshot()
                free_blocks = [b for b in snap["blocks"] if b["free"]]
                frag_metrics.append({
                    "step": step,
                    "num_free_blocks": len(free_blocks),
                    "largest_free_block_mb": max([b["size"] for b in free_blocks], default=0) / 1024**2,
                    "frag_index": snap["frag_score"]
                })

                # 3. Forward / backward
                B, T = cfg["batch_size"], cfg["seq_len"]
                x = torch.randint(0, 50257, (B, T), device="cuda")
                y = torch.randint(0, 50257, (B, T), device="cuda")
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits.view(-1, 50257), y.view(-1))
                loss.backward()
                opt.step()

                # 4. Release fragmentation anchors
                del anchors; anchors = None

                # 5. Sample utilization
                alloc_mb = torch.cuda.memory_allocated() / 1024**2
                util_samples.append(alloc_mb / total_mb * 100.0)

            except torch.cuda.OutOfMemoryError:
                oom_hit = True
                oom_step = step
                log.warning("  OOM at step %d", step)
                torch.cuda.empty_cache()
                if anchors is not None:
                    del anchors; anchors = None
                break

            finally:
                # Defrag Overhead tracking (End)
                if cb:
                    t0 = time.perf_counter()
                    try: cb.on_step_end()
                    except: pass
                    cb_overhead_s += (time.perf_counter() - t0)
                
                step_times.append(time.perf_counter() - t_step_start)

    finally:
        if cb:
            try: cb.on_train_end()
            except: pass
        del model, opt
        if anchors is not None:
            del anchors
        torch.cuda.empty_cache()

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    mean_util = statistics.mean(util_samples) if util_samples else 0.0
    total_time = sum(step_times)
    overhead_pct = (cb_overhead_s / total_time * 100.0) if total_time > 0 else 0.0

    # Aggregate frag metrics
    avg_free_blocks = statistics.mean([m["num_free_blocks"] for m in frag_metrics]) if frag_metrics else 0
    avg_frag_index = statistics.mean([m["frag_index"] for m in frag_metrics]) if frag_metrics else 0
    max_free_block = max([m["largest_free_block_mb"] for m in frag_metrics], default=0.0)

    return {
        "oom": oom_hit,
        "oom_step": oom_step,
        "peak_mb": peak_mb,
        "util_pct": mean_util,
        "overhead_pct": overhead_pct,
        "avg_free_blocks": avg_free_blocks,
        "avg_frag_index": avg_frag_index,
        "max_free_block_mb": max_free_block,
        "total_time_s": total_time
    }

# ── Mode Runner ───────────────────────────────────────────────────────────────

def run_mode(use_defrag: bool, n: int, cfg: dict) -> dict:
    tag = "DEFRAG" if use_defrag else "BASELINE"
    log.info("=" * 65)
    log.info("Mode: %s | %d trials x %d steps | Batch Size: %d", tag, n, cfg["steps"], cfg["batch_size"])
    log.info("=" * 65)
    results = []
    for i in range(n):
        log.info("[%s] Trial %d/%d ...", tag, i+1, n)
        r = run_trial(use_defrag, cfg)
        results.append(r)
        status = f"OOM@step {r['oom_step']}" if r['oom'] else "PASS"
        log.info("  -> %-12s | peak=%6.0f MB | util=%5.1f%% | overhead=%4.1f%%",
                  status, r["peak_mb"], r["util_pct"], r["overhead_pct"])

    oom_ct = sum(1 for r in results if r["oom"])
    runs_to_oom = [r["oom_step"] for r in results if r["oom"]]
    avg_runs_to_oom = statistics.mean(runs_to_oom) if runs_to_oom else cfg["steps"]

    return {
        "mode": tag, "n_trials": n,
        "oom_count": oom_ct,
        "oom_rate_pct": oom_ct / n * 100,
        "avg_runs_to_oom": avg_runs_to_oom,
        "peak_mb_mean": statistics.mean(r["peak_mb"] for r in results),
        "util_pct_mean": statistics.mean(r["util_pct"] for r in results),
        "overhead_pct_mean": statistics.mean(r["overhead_pct"] for r in results),
        "avg_free_blocks": statistics.mean(r["avg_free_blocks"] for r in results),
        "avg_frag_index": statistics.mean(r["avg_frag_index"] for r in results),
        "max_free_block_mb": statistics.mean(r["max_free_block_mb"] for r in results),
        "trials": results,
    }

# ── Stability Search ─────────────────────────────────────────────────────────

def find_max_stable_batch_size(use_defrag: bool, k: int, cfg: dict) -> int:
    """Largest batch size that runs K consecutive trials without OOM."""
    log.info("Searching for max stable batch size (K=%d, Defrag=%s)...", k, use_defrag)
    current_b = cfg["batch_size"]
    max_stable = 0
    
    # Simple linear search upwards from starting batch size
    # In a real scenario, binary search might be better, but we want to see stability.
    while current_b < 64: # Safety cap
        log.info("Testing Batch Size: %d", current_b)
        test_cfg = cfg.copy()
        test_cfg["batch_size"] = current_b
        
        failures = 0
        for i in range(k):
            r = run_trial(use_defrag, test_cfg)
            if r["oom"]:
                failures += 1
                break
        
        if failures == 0:
            max_stable = current_b
            current_b += 1
        else:
            break
            
    return max_stable

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials",      type=int, default=3)
    ap.add_argument("--steps",       type=int, default=10)
    ap.add_argument("--batch-size",  type=int, default=12)
    ap.add_argument("--seq-len",     type=int, default=512)
    ap.add_argument("--layers",      type=int, default=6)
    ap.add_argument("--frag-chunks", type=int, default=40)
    ap.add_argument("--chunk-mb",    type=int, default=12)
    ap.add_argument("--stability-k", type=int, default=2)
    ap.add_argument("--out", default="results/model_fragmentation.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log.error("CUDA not available. Exiting."); return

    gpu = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    log.info("GPU: %s | %.1f GB", gpu, total_gb)

    cfg = dict(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        layers=args.layers,
        frag_chunks=args.frag_chunks,
        chunk_mb=args.chunk_mb,
    )

    with mlflow.start_run(run_name=f"frag_modeling_{datetime.now().strftime('%m%d_%H%M')}"):
        mlflow.log_params(cfg)
        mlflow.log_param("gpu", gpu)

        # 1. Standard Comparison
        baseline = run_mode(False, args.trials, cfg)
        defrag   = run_mode(True,  args.trials, cfg)

        # 2. Stability Search
        log.info("\n--- Stability Analysis ---")
        stable_b_base = find_max_stable_batch_size(False, args.stability_k, cfg)
        stable_b_defrag = find_max_stable_batch_size(True, args.stability_k, cfg)

        # Derived Metrics
        oom_red = (baseline["oom_count"] - defrag["oom_count"]) / baseline["oom_count"] * 100 if baseline["oom_count"] > 0 else 0
        util_gain = defrag["util_pct_mean"] - baseline["util_pct_mean"]
        
        mlflow.log_metric("oom_reduction_pct", oom_red)
        mlflow.log_metric("util_gain_pp", util_gain)
        mlflow.log_metric("stable_batch_baseline", stable_b_base)
        mlflow.log_metric("stable_batch_defrag", stable_b_defrag)

        log.info("\n" + "=" * 65)
        log.info("FINAL SUMMARY")
        log.info("=" * 65)
        log.info("  OOM Rate (Base vs Defrag) : %.1f%% vs %.1f%%", baseline["oom_rate_pct"], defrag["oom_rate_pct"])
        log.info("  Avg Runs to First OOM     : %.1f vs %.1f", baseline["avg_runs_to_oom"], defrag["avg_runs_to_oom"])
        log.info("  Max Stable Batch Size     : %d vs %d", stable_b_base, stable_b_defrag)
        log.info("  Avg GPU Utilization       : %.1f%% vs %.1f%%", baseline["util_pct_mean"], defrag["util_pct_mean"])
        log.info("  Defrag Overhead           : %.2f%%", defrag["overhead_pct_mean"])
        log.info("-" * 65)
        log.info("  Avg Free Blocks           : %.1f vs %.1f", baseline["avg_free_blocks"], defrag["avg_free_blocks"])
        log.info("  Avg Frag Index            : %.3f vs %.3f", baseline["avg_frag_index"], defrag["avg_frag_index"])
        log.info("  Largest Free Block (MB)   : %.1f vs %.1f", baseline["max_free_block_mb"], defrag["max_free_block_mb"])
        log.info("=" * 65)

        out = {
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu,
            "config": cfg,
            "baseline": baseline,
            "defrag": defrag,
            "stability": {
                "baseline_max_batch": stable_b_base,
                "defrag_max_batch": stable_b_defrag,
                "k": args.stability_k
            },
            "summary": {
                "oom_reduction_pct": oom_red,
                "util_gain_pp": util_gain,
                "overhead_pct": defrag["overhead_pct_mean"]
            }
        }
        
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        
        mlflow.log_artifact(args.out)
        log.info("Full metrics saved to %s", args.out)

if __name__ == "__main__":
    main()
