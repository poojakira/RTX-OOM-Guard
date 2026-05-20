"""
run.py
======

Unified execution CLI for the rtx_oom_guard ML infrastructure.
Wraps the entire system behind a clean, config-driven pipeline.

Usage:
  python run.py --config configs/config.yaml
"""

import argparse
import sys
import os
import yaml

from rtx_oom_guard.utils import DefragConfig
from rtx_oom_guard.scheduler.monitor import DefragMonitor
from rtx_oom_guard.utils import get_logger

log = get_logger("infra_runner")

def main():
    parser = argparse.ArgumentParser(description="run.py: Config-Driven Execution")
    parser.add_argument("--config", required=True, help="Path to execution config.yaml")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        log.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    if not yaml_data or not isinstance(yaml_data, dict):
        log.error("Configuration file is empty or invalid.")
        sys.exit(1)

    # Initialize the centralized configuration
    cfg = DefragConfig()
    for k, v in yaml_data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
            
    log.info(f"Configuration loaded from {config_path}")
    log.info("-" * 40)
    log.info(f"Threshold:    {cfg.frag_threshold}")
    log.info(f"DDP Sync:     {cfg.ddp_sync}")
    log.info(f"Async Defrag: {cfg.async_compaction}")
    log.info("-" * 40)

    # Boot the scheduler daemon
    log.info("Booting ML Infrastructure Monitor...")
    monitor = DefragMonitor(config=cfg)
    monitor.start()
    
    log.info("System is active. Press Ctrl+C to terminate.")
    try:
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        log.info("Shutdown signal received.")
        monitor.stop()
        log.info("System halted.")

if __name__ == "__main__":
    main()
