"""Validation tooling: multi-solver reference matrix + EOS-level model diagnostic."""
from .reference_matrix import run as run_reference_matrix, build_ladder
from .model_diagnostic import model_diagnostic

__all__ = ["run_reference_matrix", "build_ladder", "model_diagnostic"]
