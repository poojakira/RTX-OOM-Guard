import torch
from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter

def test_defragmenter_repack_numerical_parity():
    """
    Test that GPUMemoryDefragmenter successfully coalesces scattered tensors
    into a single contiguous block without changing their values.
    """
    # Create some dummy scattered tensors
    # We use CPU tensors because CI runs on CPU, but defragmenter handles both safely.
    t1 = torch.randn(10, 10, requires_grad=True)
    t2 = torch.randn(5, 5, requires_grad=True)
    t3 = torch.randn(100, requires_grad=True)

    tensors = [t1, t2, t3]

    # Snapshot original values
    orig_t1 = t1.clone().detach()
    orig_t2 = t2.clone().detach()
    orig_t3 = t3.clone().detach()

    # The tensor memory should not be strictly contiguous together right now
    assert t1.data_ptr() != t2.data_ptr()

    # Defragment
    engine = GPUMemoryDefragmenter(use_triton=False)
    metrics = engine.defragment_tensors(tensors)

    assert not metrics.get("skipped", False)
    assert metrics["tensors_compacted"] == 3

    # Verify values are perfectly identical
    assert torch.allclose(t1, orig_t1)
    assert torch.allclose(t2, orig_t2)
    assert torch.allclose(t3, orig_t3)

    # Verify autograd is preserved
    assert t1.requires_grad
    assert t2.requires_grad
    assert t3.requires_grad

    # Compute a dummy loss to ensure graph runs
    loss = t1.sum() + t2.sum() + t3.sum()
    loss.backward()

    # Gradients should have populated
    assert t1.grad is not None
    assert t2.grad is not None
    assert t3.grad is not None

def test_defragmenter_empty_input():
    engine = GPUMemoryDefragmenter()
    metrics = engine.defragment_tensors([])
    assert metrics.get("skipped")
