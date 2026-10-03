"""JAX physics of NSCool's cooling runs (paper 7, step C2).

Each process follows NSCool's own routine, the reference, and is tested
zone by zone against ``NSCool.zone_dump``.  Inputs are per-zone arrays,
and every function works under jit, vmap and grad.  Importing this
package turns on JAX double precision.
"""
import jax

jax.config.update("jax_enable_x64", True)

from . import core_nu, crust_nu, gaps, heat  # noqa: E402

__all__ = ["core_nu", "crust_nu", "gaps", "heat"]
