"""
scripts/generate_senior_dataset.py
====================================
Batch-generates a diverse set of GPU memory traces using the WorkloadSimulator.

Produces 100+ Parquet files in data/traces/senior_v1/ with permuted
hyperparameters (batch size, seq length, model architecture, VRAM limits)
and risk labels (stable / high_risk / critical).

Usage::

    python scripts/generate_senior_dataset.py
    python scripts/generate_senior_dataset.py --output data/traces/senior_v1 --count 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.workload_simulator import (
    GPUWorkload,
    TransformerSpec,
    CNNSpec,
)

# Configurations matrix

def _build_configs() -> list[dict]:
    """Build a matrix of workload configurations for diversity."""
    configs = []

    for arch_fn, name in [
        (TransformerSpec.gpt2, "gpt2"),
        (TransformerSpec.gpt2_medium, "gpt2m"),
        (TransformerSpec.bert_base, "bert_base"),
        (TransformerSpec.bert_large, "bert_large"),
        (TransformerSpec.vit_large, "vit_large"),
    ]:
        for bs in [2, 4, 8, 16]:
            for seq in [128, 256, 512, 1024]:
                try:
                    spec = arch_fn(batch_size=bs, seq_len=seq)
                except TypeError:
                    spec = arch_fn(batch_size=bs)

                # Vary VRAM to create different risk profiles
                for vram_mb in [6144, 8192, 12288]:
                    configs.append({
                        "spec": spec,
                        "name": f"{name}_bs{bs}_seq{seq}_vram{vram_mb}",
                        "vram_mb": vram_mb,
                        "steps": 200,
                    })

    for arch_fn, name in [
        (CNNSpec.resnet50, "resnet50"),
        (CNNSpec.resnet101, "resnet101"),
        (CNNSpec.efficientnet, "effnet_b4"),
    ]:
        for bs in [8, 16, 32, 64]:
            spec = arch_fn(batch_size=bs)
            for vram_mb in [6144, 8192]:
                configs.append({
                    "spec": spec,
                    "name": f"{name}_bs{bs}_vram{vram_mb}",
                    "vram_mb": vram_mb,
                    "steps": 200,
                })

    return configs


def _risk_label(events: list[dict]) -> str:
    """Classify a trace as stable / high_risk / critical."""
    ooms = sum(1 for e in events if e.get("oom", False))
    frags = [e["fragmentation"] for e in events]
    max_frag = max(frags) if frags else 0
    mean_frag = float(np.mean(frags)) if frags else 0

    if ooms > 5 or max_frag > 0.7:
        return "critical"
    elif ooms > 0 or mean_frag > 0.3:
        return "high_risk"
    else:
        return "stable"


# Main

def main():
    ap = argparse.ArgumentParser(description="Generate senior GPU memory trace dataset")
    ap.add_argument("--output", default="data/traces/senior_v1",
                    help="Output directory for Parquet files")
    ap.add_argument("--count", type=int, default=120,
                    help="Max number of trace files to generate")
    ap.add_argument("--steps", type=int, default=200,
                    help="Default training steps per trace")
    args = ap.parse_args()

    out_dir = Path(ROOT / args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = _build_configs()
    # Shuffle and cap to requested count
    rng = np.random.RandomState(2024)
    indices = rng.permutation(len(configs))[:args.count]

    manifest = []
    generated = 0

    print(f"Generating {len(indices)} traces in {out_dir} ...")

    for i, idx in enumerate(indices):
        cfg = configs[idx]
        spec = cfg["spec"]
        name = cfg["name"]
        vram = cfg["vram_mb"]
        steps = cfg.get("steps", args.steps)

        try:
            wl = GPUWorkload(
                spec=spec,
                vram_mb=vram,
                noise_std=0.03,
                cache_clear_interval=50,
            )
            events = wl.run(steps=steps, seed=42 + i)
        except Exception as e:
            print(f"  [{i+1:3d}] SKIP {name}: {e}")
            continue

        if len(events) < 100:
            continue

        # Save Parquet
        df = pd.DataFrame(events)
        fname = f"{name}.parquet"
        df.to_parquet(out_dir / fname, engine="pyarrow", index=False)

        label = _risk_label(events)
        frags = [e["fragmentation"] for e in events]
        ooms = sum(1 for e in events if e.get("oom", False))

        manifest.append({
            "file": fname,
            "architecture": spec.name,
            "events": len(events),
            "ooms": ooms,
            "mean_frag": round(float(np.mean(frags)), 4),
            "max_frag": round(max(frags), 4),
            "risk_label": label,
            "vram_mb": vram,
        })

        generated += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(indices):
            print(f"  [{i+1:3d}/{len(indices)}] {name:40s}  "
                  f"events={len(events):5d}  frag={np.mean(frags):.3f}  "
                  f"OOMs={ooms}  label={label}")

    # Save manifest
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary stats
    labels = [m["risk_label"] for m in manifest]
    print(f"\n{'='*60}")
    print(f"Generated {generated} traces → {out_dir}")
    print(f"  stable:    {labels.count('stable')}")
    print(f"  high_risk: {labels.count('high_risk')}")
    print(f"  critical:  {labels.count('critical')}")
    print(f"  manifest:  {manifest_path}")


if __name__ == "__main__":
    main()
