"""Run geometry from a checkout with an explicitly chosen Python interpreter.

The active interpreter's packages take precedence. Existing .venv packages
provide a fallback for pure Python dependencies such as PyYAML.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
venv_packages = root / ".venv" / "Lib" / "site-packages"
if venv_packages.is_dir():
    sys.path.append(str(venv_packages))

from bounding_box.geometry import main

if __name__ == "__main__":
    main()
