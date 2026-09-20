"""Make the runner's modules importable the way ``run/rollout.py`` imports them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "run"))
