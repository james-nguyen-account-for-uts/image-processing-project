"""Run without installing the local package (dependencies are still required)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fall_detection.image_evaluation import main

if __name__ == '__main__':
    main()
