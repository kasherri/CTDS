"""
Cell-Type Dynamical Systems (CTDS).

A JAX-native implementation for modeling neural population dynamics
with biologically-constrained connectivity and Dale's law.
"""

__version__ = "0.1.0"
__author__ = "Njeri Njoroge"
__email__ = "njerinjoroge@gmail.com"

# Import main classes and functions
from .models import CTDS, BaseCTDS
from .params import (
    ParamsCTDS,
    ParamsCTDSDynamics, 
    ParamsCTDSEmissions,
    ParamsCTDSInitial,
    ParamsCTDSConstraints,
    SufficientStats,
)
from .inference import InferenceBackend, DynamaxLGSSMBackend

# Import key utilities
from .utils import (
    solve_dale_QP,
    solve_constrained_QP,
    compute_sufficient_statistics,
    estimate_J,
    blockwise_NMF,
)

__all__ = [
    # Main classes
    "CTDS",
    "BaseCTDS",
    # Parameters
    "ParamsCTDS",
    "ParamsCTDSDynamics",
    "ParamsCTDSEmissions", 
    "ParamsCTDSInitial",
    "ParamsCTDSConstraints",
    "SufficientStats",
    # Inference
    "InferenceBackend", 
    "DynamaxLGSSMBackend",
    # Utilities
    "solve_dale_QP",
    "solve_constrained_QP", 
    "compute_sufficient_statistics",
    "estimate_J",
    "blockwise_NMF",
]
