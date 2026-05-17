"""
scripts/generate_large_dataset.py
===================================
Generate a LARGE-SCALE GPU memory trace dataset (billions of events).

Uses chunked streaming to Parquet to avoid OOM on the generation machine.
Each trace runs for 100K-500K steps, producing 1M-5M events per file.
Across 500+ traces this produces billions of total data points.

Usage::

    python scripts/generate_large_dataset.py
    python scripts/generate_large_dataset.py --traces 500 --steps 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.workload_simulator import (
    GPUWorkload,
    TransformerSpec,
    CNNSpec,
)


# Streamed workload runner — writes Parquet in chunks

CHUNK_SIZE = 50_000  # Flush to disk every 50K events


def _run_streamed(
    spec,
    vram_mb: float,
    steps: int,
    seed: int,
    output_path: Path,
    noise_std: float = 0.03,
    cache_clear_interval: int = 100,
) -> dict:
    """
    Run workload and stream events to Parquet in chunks.
    Returns summary metadata.
    """
    wl = GPUWorkload(
        spec=spec,
        vram_mb=vram_mb,
        noise_std=noise_std,
        cache_clear_interval=cache_clear_interval,
    )

    # Override run() to do chunked writes
    import random
    rng = np.random.RandomState(seed)
    random.seed(seed)

    # Initialize model params + optimizer (same as GPUWorkload.run)
    if isinstance(spec, TransformerSpec):
        per_layer = spec.param_mb / max(spec.layers, 1)
        for layer_i in range(spec.layers):
            bid = wl._alloc_or_oom(per_layer, "param", 0, "init")
            if bid is not None:
                wl._param_blocks.append(bid)
    else:
        bid = wl._alloc_or_oom(spec.param_mb, "param", 0, "init")
        if bid is not None:
            wl._param_blocks.append(bid)

    opt_bid = wl._alloc_or_oom(spec.optimizer_state_mb, "optimizer", 0, "init")
    if opt_bid is not None:
        wl._optimizer_blocks.append(opt_bid)

    # Schema for parquet
    schema = pa.schema([
        ("timestamp_ns", pa.int64()),
        ("step", pa.int32()),
        ("phase", pa.string()),
        ("action", pa.int8()),
        ("delta_bytes", pa.int64()),
        ("abs_allocated", pa.float32()),
        ("abs_reserved", pa.float32()),
        ("fragmentation", pa.float32()),
        ("utilization", pa.float32()),
        ("tag", pa.string()),
        ("oom", pa.bool_()),
    ])

    writer = pq.ParquetWriter(str(output_path), schema, compression="snappy")
    total_events = 0
    total_ooms = 0
    all_frags = []

    buffer = []

    def _flush():
        nonlocal total_events
        if not buffer:
            return
        df = pd.DataFrame(buffer)
        # Cast types to match schema
        df["step"] = df["step"].astype("int32")
        df["action"] = df["action"].astype("int8")
        df["abs_allocated"] = df["abs_allocated"].astype("float32")
        df["abs_reserved"] = df["abs_reserved"].astype("float32")
        df["fragmentation"] = df["fragmentation"].astype("float32")
        df["utilization"] = df["utilization"].astype("float32")
        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        writer.write_table(table)
        total_events += len(buffer)
        buffer.clear()

    # Training loop
    for step in range(1, steps + 1):
        activation_ids = []
        gradient_ids = []

        # ── Forward ──
        n_act = max(1, spec.layers // 2)
        for i in range(n_act):
            act_size = spec.activation_per_layer_mb * (1 + rng.normal(0, 0.05))
            bid = wl._alloc_or_oom(max(act_size, 0.5), "activation", step, "forward")
            if bid is not None:
                activation_ids.append(bid)

        n_temps = rng.randint(2, 8)
        temp_ids = []
        for _ in range(n_temps):
            temp_size = rng.uniform(0.5, 4.0)
            bid = wl._alloc_or_oom(temp_size, "temp", step, "forward")
            if bid is not None:
                temp_ids.append(bid)

        for tid in temp_ids:
            freed = wl.allocator.free(tid)
            wl._emit(step, "forward", 0, freed, "temp")

        # ── Backward ──
        grad_size = spec.gradient_mb / max(spec.layers, 1)
        for i in range(max(1, spec.layers // 3)):
            bid = wl._alloc_or_oom(
                grad_size * (1 + rng.normal(0, 0.03)), "gradient", step, "backward"
            )
            if bid is not None:
                gradient_ids.append(bid)

        for aid in activation_ids:
            freed = wl.allocator.free(aid)
            wl._emit(step, "backward", 0, freed, "activation")

        # ── Optimizer ──
        for _ in range(rng.randint(1, 4)):
            temp_size = rng.uniform(1.0, 6.0)
            bid = wl._alloc_or_oom(temp_size, "optimizer_temp", step, "optimizer")
            if bid is not None:
                freed = wl.allocator.free(bid)
                wl._emit(step, "optimizer", 0, freed, "optimizer_temp")

        # ── Cleanup ──
        for gid in gradient_ids:
            freed = wl.allocator.free(gid)
            wl._emit(step, "cleanup", 0, freed, "gradient")

        if cache_clear_interval > 0 and step % cache_clear_interval == 0:
            cleared = wl.allocator.empty_cache()
            if cleared > 0:
                wl._emit(step, "cleanup", 0, cleared, "cache_clear")

        # Drain events buffer into our local buffer
        for evt in wl.events:
            d = asdict(evt)
            buffer.append(d)
            all_frags.append(d["fragmentation"])
            if d["oom"]:
                total_ooms += 1
        wl.events.clear()

        # Flush chunk
        if len(buffer) >= CHUNK_SIZE:
            _flush()

    # Final flush
    _flush()
    writer.close()

    mean_frag = float(np.mean(all_frags)) if all_frags else 0.0
    max_frag = float(np.max(all_frags)) if all_frags else 0.0

    return {
        "events": total_events,
        "ooms": total_ooms,
        "mean_frag": round(mean_frag, 4),
        "max_frag": round(max_frag, 4),
    }


# Config matrix

def _build_large_configs(steps: int):
    configs = []

    transformer_archs = [
        ("gpt2", TransformerSpec.gpt2),
        ("gpt2m", TransformerSpec.gpt2_medium),
        ("bert_base", TransformerSpec.bert_base),
        ("bert_large", TransformerSpec.bert_large),
        ("vit_large", TransformerSpec.vit_large),
        ("llama7b", TransformerSpec.llama_7b),
    ]

    cnn_archs = [
        ("resnet50", CNNSpec.resnet50),
        ("resnet101", CNNSpec.resnet101),
        ("effnet", CNNSpec.efficientnet),
    ]

    for arch_name, arch_fn in transformer_archs:
        for bs in [2, 4, 8, 16, 32]:
            for seq in [128, 256, 512, 1024, 2048]:
                try:
                    spec = arch_fn(batch_size=bs, seq_len=seq)
                except TypeError:
                    spec = arch_fn(batch_size=bs)
                for vram in [4096, 6144, 8192, 12288, 16384, 24576]:
                    configs.append({
                        "spec": spec,
                        "name": f"{arch_name}_bs{bs}_seq{seq}_vram{vram}",
                        "vram_mb": vram,
                        "steps": steps,
                    })

    for arch_name, arch_fn in cnn_archs:
        for bs in [4, 8, 16, 32, 64, 128]:
            spec = arch_fn(batch_size=bs)
            for vram in [4096, 8192, 16384, 24576]:
                configs.append({
                    "spec": spec,
                    "name": f"{arch_name}_bs{bs}_vram{vram}",
                    "vram_mb": vram,
                    "steps": steps,
                })

    return configs


def _risk_label(meta: dict) -> str:
    if meta["ooms"] > 50 or meta["max_frag"] > 0.7:
        return "critical"
    elif meta["ooms"] > 0 or meta["mean_frag"] > 0.3:
        return "high_risk"
    else:
        return "stable"


# Main

def main():
    ap = argparse.ArgumentParser(
        description="Generate LARGE-SCALE GPU memory trace dataset"
    )
    ap.add_argument("--output", default="data/traces/large_v1")
    ap.add_argument("--traces", type=int, default=500,
                    help="Number of traces to generate")
    ap.add_argument("--steps", type=int, default=200_000,
                    help="Training steps per trace")
    args = ap.parse_args()

    out_dir = Path(ROOT / args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = _build_large_configs(args.steps)
    rng = np.random.RandomState(2024)
    indices = rng.permutation(len(configs))[:args.traces]

    manifest = []
    grand_total_events = 0
    t0 = time.time()

    print(f"Generating {len(indices)} large traces ({args.steps:,} steps each) → {out_dir}")
    print(f"Config space: {len(configs)} total configurations")
    print(f"Estimated events: ~{len(indices) * args.steps * 15:,}")
    print()

    for i, idx in enumerate(indices):
        cfg = configs[idx]
        spec = cfg["spec"]
        name = cfg["name"]
        vram = cfg["vram_mb"]
        steps = cfg["steps"]

        trace_path = out_dir / f"{name}.parquet"
        t_trace = time.time()

        try:
            meta = _run_streamed(
                spec=spec,
                vram_mb=vram,
                steps=steps,
                seed=42 + i,
                output_path=trace_path,
            )
        except Exception as e:
            print(f"  [{i+1:3d}] SKIP {name}: {e}")
            continue

        elapsed_trace = time.time() - t_trace
        label = _risk_label(meta)
        grand_total_events += meta["events"]

        entry = {
            "file": f"{name}.parquet",
            "architecture": spec.name,
            "steps": steps,
            "events": meta["events"],
            "ooms": meta["ooms"],
            "mean_frag": meta["mean_frag"],
            "max_frag": meta["max_frag"],
            "risk_label": label,
            "vram_mb": vram,
            "generation_time_s": round(elapsed_trace, 1),
        }
        manifest.append(entry)

        if (i + 1) % 10 == 0 or (i + 1) == len(indices):
            elapsed_total = time.time() - t0
            rate = grand_total_events / elapsed_total if elapsed_total > 0 else 0
            print(
                f"  [{i+1:3d}/{len(indices)}] {name:45s}  "
                f"events={meta['events']:>8,}  frag={meta['mean_frag']:.3f}  "
                f"OOMs={meta['ooms']:>4}  {label:10s}  "
                f"[{elapsed_trace:.1f}s]  "
                f"total={grand_total_events:>12,} @ {rate:,.0f} evt/s"
            )

    elapsed = time.time() - t0

    # Save manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    labels = [m["risk_label"] for m in manifest]
    total_size_mb = sum(f.stat().st_size for f in out_dir.glob("*.parquet")) / (1024**2)

    print(f"\n{'='*70}")
    print(f"Dataset generation complete in {elapsed/60:.1f} minutes")
    print(f"  Traces:       {len(manifest)}")
    print(f"  Total events: {grand_total_events:,}")
    print(f"  Total size:   {total_size_mb:,.1f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"  stable:       {labels.count('stable')}")
    print(f"  high_risk:    {labels.count('high_risk')}")
    print(f"  critical:     {labels.count('critical')}")
    print(f"  Manifest:     {manifest_path}")
    print(f"  Throughput:   {grand_total_events/elapsed:,.0f} events/sec")


if __name__ == "__main__":
    main()
