"""Add src/ to sys.path so tests can import top-level packages without
needing `pip install -e .`. Pytest auto-detects this file at repo root.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
