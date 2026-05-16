# 🛡️ rtx-oom-guard Security Policy

rtx-oom-guard is an infrastructure-level memory manager that performs physical VRAM repacking. Secure operations are paramount for enterprise clusters.

---

## 🔒 Security Practices

1.  **Memory Isolation**: Triton kernels use raw physical address management but are constrained by individual PyTorch CUDA Streams. This ensures that memory migrations cannot overwrite other active CUDA kernels or standard system memory.
2.  **Kernel Integrity**: We only execute predefined Triton kernels (compaction_copy, fragmentation_scan). No arbitrary code execution is permitted on the GPU via rtx-oom-guard.
3.  **Encapsulated Telemetry**: The FastAPI server (Telemetry Surface) uses read-only snapshots and JSON telemetry files. It does NOT possess write-access to the GPU or the training process memory pointers.

---

## 💥 Blast-Radius Considerations

- **NCCL Distributed Barriers**: If a compaction event fails on Rank N in a multi-GPU DDP training run, the `DDPSyncManager` ensures that other ranks are NOT deadlocked. We gracefully time out and resume training with the original (fragmented) memory state.
- **Graceful Fallbacks**: If Triton kernels are unavailable or fail validation, the system falls back to `torch.clone()` — a standard, secure PyTorch operation.

---

## 🧪 Vulnerability Reporting

If you discover a security vulnerability in **rtx-oom-guard** (e.g., potential data corruption during repacking, memory leak, or API exploit), please send an email to the maintainers:

**Security Contact**: pooja.kiran@rtx-oom-guard.ai

We aim to respond to critical security reports within **24 hours**. Please do NOT open public issues for security vulnerabilities until we have discussed a coordinated disclosure.

---

## 📄 Compliance

rtx-oom-guard is intended for research and production-readiness prototyping. We recommend performing a system-wide security audit before deploying to business-critical environments.
