import os
import time
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# rtx-oom-guard Imports
from rtx_oom_guard.trainer.auto_instrument import auto_instrument
from rtx_oom_guard.trainer.ddp import DDPSyncManager
from rtx_oom_guard.utils import get_logger

# Optional MLflow
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

log = get_logger("rtx_oom_guard.train_ddp")

# Dummy Dataset & Model

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class SimpleNet(nn.Module):
    def __init__(self, size):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(size, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        return self.fc2(self.relu(self.fc(x)))

# Training Loop

def train(args):
    # DDP Setup
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Model & Optimizer
    model = SimpleNet(args.input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # rtx-oom-guard Core Instrumentation
    # Automatically inserts fragmentation monitoring and proactive compaction hooks
    model, optimizer = auto_instrument(
        model, 
        optimizer, 
        risk_threshold=args.risk_threshold,
        enabled=True
    )

    # DDP Wrapper
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)

    # Data Loading
    dataset = RandomDataset(args.input_size, args.num_samples)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # MLflow tracking
    if local_rank == 0 and HAS_MLFLOW:
        mlflow.set_experiment("rtx-oom-guard_DDP_Training")
        mlflow.start_run(run_name=f"rank_0_{int(time.time())}")
        mlflow.log_params({
            "batch_size": args.batch_size,
            "risk_threshold": args.risk_threshold,
            "world_size": dist.get_world_size(),
        })

    # DDPSyncManager for infrastructure observability
    sync_manager = DDPSyncManager()

    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for i, data in enumerate(dataloader):
            data = data.to(device)
            target = torch.randn(data.size(0), 10).to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()

            if i % 10 == 0 and local_rank == 0:
                log.info(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
                
                if HAS_MLFLOW:
                    mlflow.log_metric("loss", loss.item(), step=i)
                    # Hypothetical peak VRAM metric
                    if torch.cuda.is_available():
                        vram = torch.cuda.max_memory_reserved() / (1024**2)
                        mlflow.log_metric("peak_vram_mb", vram, step=i)

            # Infrastructure observability: Check if compaction sync is working
            # In a real run, auto_instrument handles this, but we can audit sync status
            if i % 50 == 0:
                stats = sync_manager.get_sync_status()
                if local_rank == 0:
                    log.info(f"DDP Sync Status: {stats}")

    if local_rank == 0 and HAS_MLFLOW:
        mlflow.end_run()

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--risk-threshold", type=float, default=0.8)
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
