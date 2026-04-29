"""
ctds — Cell-Type Dynamical Systems

A JAX/Dynamax-based package for fitting structured linear dynamical systems
to neural population data with cell-type and Dale's Law constraints.
"""

from .models import CTDS
from .params import ParamsCTDS, ParamsCTDSDynamics, ParamsCTDSEmissions, ParamsCTDSInitial, ParamsCTDSConstraints
from .inference import DynamaxLGSSMBackend

__version__ = "0.1.0"

__all__ = [
    "CTDS",
    "ParamsCTDS",
    "ParamsCTDSDynamics",
    "ParamsCTDSEmissions",
    "ParamsCTDSInitial",
    "ParamsCTDSConstraints",
    "DynamaxLGSSMBackend",
]
