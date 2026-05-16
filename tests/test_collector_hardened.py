import torch
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from rtx_oom_guard.profiler.collector import AllocationCollector, collect_from_model
from rtx_oom_guard.utils import DefragConfig

def test_collector_circular_buffer():
    """Verify that the collector respects max_events and clears old ones."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=123), \
         patch("torch.cuda.memory_reserved", return_value=123):
        config = DefragConfig(max_events=5)
        collector = AllocationCollector(config)
    
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", side_effect=range(0, 700, 100)), \
         patch("torch.cuda.memory_reserved", return_value=1000):
        
        # Record 7 events (max is 5)
        for _ in range(7):
            collector.record()
            
        assert collector.event_count == 5
        df = collector.to_dataframe()
        assert len(df) == 5

@pytest.mark.asyncio
async def test_collector_polling_lifecycle():
    """Verify collector starts and stops polling thread."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", return_value=1024), \
         patch("torch.cuda.memory_reserved", return_value=2048):
        config = DefragConfig(poll_interval_ms=1)
        collector = AllocationCollector(config)
        
        collector.start()
        assert collector._active == True
        assert collector._thread is not None
        assert collector._thread.is_alive()
        
        import asyncio
        await asyncio.sleep(0.05)
        
        collector.stop()
        assert collector._active == False
        assert not collector._thread.is_alive()

def test_collector_save_load(tmp_path):
    """Verify Parquet export with mock events."""
    collector = AllocationCollector()
    collector._events = [
        {"timestamp_ns": 100, "delta_bytes": 1024, "action": 1, "abs_allocated": 1024, "abs_reserved": 2048, "fragmentation": 0.5},
        {"timestamp_ns": 200, "delta_bytes": -512, "action": 0, "abs_allocated": 512, "abs_reserved": 2048, "fragmentation": 0.75},
    ]
    
    path = tmp_path / "test_trace.parquet"
    with patch("rtx_oom_guard.profiler.collector.log"):
        collector.save(str(path))
    
    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 2

def test_collect_from_model_mocked():
    """Verify end-to-end model collection orchestration."""
    # We use a cycling iterator for memory_allocated to avoid StopIteration
    from itertools import count
    
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.memory_allocated", side_effect=count(step=100)), \
         patch("torch.cuda.memory_reserved", return_value=10000), \
         patch("torch.cuda.synchronize"), \
         patch("rtx_oom_guard.trainer._models.build_gpt2") as mock_build:
        
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        # Model output must require grad for backward() to work
        loss_tensor = torch.tensor(1.0, requires_grad=True)
        mock_model.return_value = loss_tensor
        mock_build.return_value = (mock_model, torch.tensor([1, 2]))
        
        count_val = collect_from_model("gpt2", iterations=2)
        assert count_val > 0
        assert mock_build.called

def test_collect_from_model_invalid_name():
    """Verify error for unknown model name."""
    with patch("torch.cuda.is_available", return_value=True):
        with pytest.raises(ValueError, match="Unknown model"):
            collect_from_model("invalid_model")
