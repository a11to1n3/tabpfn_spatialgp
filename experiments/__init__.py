"""The paper's experiments. Run the modules from the repository root: python -m experiments.<stage>.<module>."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))    # the package, as in this repository
