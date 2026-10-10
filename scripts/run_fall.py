"""Run fall-train/evaluate/detect from a checkout with a chosen interpreter."""

import argparse
import importlib
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
venv_packages = root / ".venv" / "Lib" / "site-packages"
if venv_packages.is_dir():
    # Active interpreter dependencies take precedence; PyYAML can use its
    # existing pure Python installation when the project venv is unavailable.
    sys.path.append(str(venv_packages))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("command", choices=["train", "evaluate", "detect"])
parser.add_argument("arguments", nargs=argparse.REMAINDER)
args = parser.parse_args()
sys.argv = [f"fall-{args.command}", *args.arguments]
importlib.import_module(f"fall_detection.{args.command}").main()
