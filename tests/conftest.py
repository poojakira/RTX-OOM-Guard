"""
tests/conftest.py — Shared test configuration and fixtures.

Ensures the project root is on sys.path so tests can import rtx_oom_guard
and scripts without per-file path manipulation.
"""

import os
import sys

import pytest

# Ensure project root is importable for all tests
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Provide a temporary results directory for telemetry output."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir
