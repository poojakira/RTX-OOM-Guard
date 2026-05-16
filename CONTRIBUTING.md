# 🤝 Contributing to rtx-oom-guard

We welcome contributions from the community to enhance the stability, performance, and observability of **rtx-oom-guard**.

---

## 🛠️ Local Development Environment

1.  **Clone Package**:
    ```bash
    git clone https://github.com/poojakira/Predictive-GPU-Memory-Defragmenter.git
    cd Predictive-GPU-Memory-Defragmenter
    ```
2.  **Install Devo Dependencies**:
    ```bash
    pip install -e ".[dev]"
    ```
3.  **Run Tests**:
    ```bash
    pytest tests/
    ```

---

## 🏗️ Pull Request (PR) Workflow

1.  **Branching**: Create a feature branch from `main`.
2.  **Coding Standards**:
    - Use `black` for formatting.
    - Use `ruff` for linting.
    - Annotate with type hints (MyPy).
3.  **Tests**: All PRs MUST include relevant unit or integration tests in `tests/`.
4.  **Documentation**: Update `README.md` or `benchmarks.md` if your change impacts performance metrics or APIs.

---

## 🔬 Benchmark Verification

If your change modifies the **Triton Kernels** or the **Defragmenter Engine**, you MUST run the benchmark suite and provide the results in the PR description:

```bash
python run_benchmark.py --steps 500
```

---

## 💬 Issue Reporting

- **Bug Reports**: Provide a minimal reproduction script and the `results/live_telemetry.json` (if applicable).
- **Feature Requests**: Describe the business use case and the expected impact on VRAM efficiency.

---

## 🎓 Code of Conduct

Be respectful and professional. rtx-oom-guard is a technical project aimed at improving GPU efficiency for the entire community.
