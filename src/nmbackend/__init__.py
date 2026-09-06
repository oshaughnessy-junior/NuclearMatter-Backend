"""NuclearMatter-Backend (nmbackend).

A differentiable (JAX) neutron-star EOS + TOV toolkit for EOS inference, designed
to interface with RIFT (per-EOS evidence / hyperpipeline) and gwkokab (population
inference).  See README.md and docs/architecture.md.
"""
from . import constants
from . import ejecta
from . import ipad
from . import postmerger
# lab_likelihood, priors, inference, rift, adapters are add-ons imported
# explicitly (from nmbackend import lab_likelihood), not eagerly here.
from .eos import MetaModelEOS, PolytropeEOS, EOSModel, EOSTable
from .tov import solve_structure, build_family, m_max

__version__ = "0.0.1"

__all__ = [
    "constants", "ejecta",
    "MetaModelEOS", "PolytropeEOS", "EOSModel", "EOSTable",
    "solve_structure", "build_family", "m_max",
]
