"""Shared test setup: the repository root (tests run without GPUs, simulators or network)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "rl"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
