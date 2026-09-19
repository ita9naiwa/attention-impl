"""Run a script or pytest with this checkout's CuTe namespace."""

import runpy
import sys
from pathlib import Path

import flash_attn

root = Path(__file__).resolve().parents[1]
flash_attn.__path__ = [str(root / "flash_attn"), *flash_attn.__path__]
sys.argv = sys.argv[1:]
if sys.argv[0] == "-m":
    sys.argv = sys.argv[1:]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
else:
    runpy.run_path(sys.argv[0], run_name="__main__")
