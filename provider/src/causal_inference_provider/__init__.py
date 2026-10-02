"""Offline causal estimation, conditional on explicitly declared assumptions."""

from .feasibility import build_train_only_feasibility_report
from .provider import CausalInferenceProvider

__all__ = ["CausalInferenceProvider", "build_train_only_feasibility_report"]
