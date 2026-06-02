"""NuclearMatter-Backend quickstart.

    PYTHONPATH=src python examples/quickstart.py

Shows: build a physical (metamodel) EOS, solve the TOV family, read off
M_max / R(1.4) / Lambda(1.4), take an AD sensitivity, and wrap it for RIFT.
"""
import jax
import numpy as np

from nmbackend.eos import MetaModelEOS
from nmbackend.tov import build_family, m_max, radius_of_mass, lambda_of_mass
from nmbackend.tov.solver import solve_structure
from nmbackend.adapters import EOSConcreteNMB

mm = MetaModelEOS()

# 1) a physical EOS from nuclear empirical parameters (+ stiffness knob)
params = dict(K_sat=230.0, E_sym=32.0, L_sym=60.0, cs2_tail=0.4)
table = mm.eos_table(params)
fam = build_family(table, n_steps=800)
print(f"M_max   = {float(m_max(fam)):.3f} Msun")
print(f"R(1.4)  = {float(radius_of_mass(fam, 1.4)):.2f} km")
print(f"L(1.4)  = {float(lambda_of_mass(fam, 1.4)):.1f}")

# 2) forward-mode AD sensitivity (the delta-TOV / Fisher building block)
hc = 0.20 * float(table.h[-1])           # ~1.4 Msun central enthalpy
dR_dLsym = jax.jacfwd(
    lambda L: solve_structure(mm.eos_table({**params, "L_sym": L}), hc,
                              n_steps=800).R_km
)(60.0)
print(f"dR/dL_sym = {float(dR_dLsym):.4f} km / MeV")

# 3) drop into RIFT (EOSConcrete-compatible)
eos_rift = EOSConcreteNMB(mm, params, name="MM-Lsym60")
print(eos_rift)
print(f"RIFT lambda_from_m(1.4) = {eos_rift.lambda_from_m(1.4):.1f}")
