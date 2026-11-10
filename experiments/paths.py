"""Folders of the repository."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"          # the GP's prior, and the comparators' stored settings, versions and factors
