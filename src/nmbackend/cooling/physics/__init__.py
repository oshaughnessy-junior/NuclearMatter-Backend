"""JAX physics of NSCool's cooling runs (paper 7, step C2).

Each process follows NSCool's own routine, the reference, and is tested
zone by zone against ``NSCool.zone_dump``.  Inputs are per-zone arrays,
and every function works under jit, vmap and grad.  Importing this
package turns on JAX double precision.  README.md lists the processes
and where NSCool departs from its cited papers.
"""
import jax

jax.config.update("jax_enable_x64", True)

from . import conduct, core_conduct, core_nu, crust_nu, envelope, gaps, heat  # noqa: E402
from .curve import isothermal_curve  # noqa: E402
from .star import IsoSums, Star, isothermal_sums, zone_terms  # noqa: E402

__all__ = ["IsoSums", "Star", "conduct", "core_conduct", "core_nu", "crust_nu", "envelope", "gaps",
           "heat", "isothermal_curve", "isothermal_sums", "zone_terms"]
