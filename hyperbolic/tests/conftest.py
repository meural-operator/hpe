"""Pytest configuration: add the project root to sys.path so test modules can
import `model`, `math_utils`, `loss`, etc. without installing the package."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
