"""rtx_oom_guard.defrag_engine — GPU memory defragmentation engine."""

from rtx_oom_guard.defrag_engine.defragmenter import GPUMemoryDefragmenter
from rtx_oom_guard.defrag_engine.compactor import MemoryCompactor
from rtx_oom_guard.defrag_engine.policy import MitigationPolicy

__all__ = ["GPUMemoryDefragmenter", "MemoryCompactor", "MitigationPolicy"]
