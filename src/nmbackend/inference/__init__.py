"""Inference utilities built on the differentiable EOS/TOV core.

Currently: forward-mode sensitivity (Jacobian) and Fisher-matrix forecasting of NS
observables (Lambda(m), R(m), M_max) with respect to EOS parameters -- the first
consumer of the backend's differentiability.
"""
from .fisher import (eos_observables, sensitivity_matrix, fisher_matrix,
                     fisher_covariance, GRADIENT_RTOL, GRADIENT_ATOL)

__all__ = ["eos_observables", "sensitivity_matrix", "fisher_matrix",
           "fisher_covariance", "GRADIENT_RTOL", "GRADIENT_ATOL"]
