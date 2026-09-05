#!/usr/bin/env python3
"""Demo: the bridge "mmapipe" segment -- NEPs -> EOS -> ejecta conditions.

For a GW170817-like binary, evaluates the ejecta-conditions surrogate through
the metamodel + TOV stack at a few values of L_sym, printing every quantity
with its two systematics (fit residual scatter, inter-formula spread) and the
per-family values behind the spread, then the AD gradient d M_ej / d L_sym.

Host note (CIT): run with JAX_ENABLE_X64=1 OMP_NUM_THREADS=1, and taskset on
the big head nodes (JAX sizes its thread pool from the visible CPU count):

    JAX_ENABLE_X64=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
        taskset -c 0-15 python scripts/ejecta_demo.py
"""
import jax

from nmbackend.eos import MetaModelEOS
from nmbackend.tov import build_family
from nmbackend.tov.family import default_hc_grid
from nmbackend.ejecta import ejecta_from_family

M1, M2 = 1.35, 1.30


def family_of(L_sym, mm=MetaModelEOS()):
    t = mm.eos_table({"L_sym": L_sym})
    return build_family(t, default_hc_grid(t, n_models=140),
                        rtol=1e-9, atol=1e-11, n_steps=800)


def show(name, uq, unit=""):
    print(f"  {name:14s} = {float(uq.value):9.4g}"
          f"  +/- {float(uq.sigma_fit):8.3g} (fit)"
          f"  +/- {float(uq.sigma_formula):8.3g} (formula)   {unit}")


def main():
    for L in (40.0, 60.0, 80.0):
        ec = ejecta_from_family(family_of(L), M1, M2)
        print(f"\nL_sym = {L:.0f} MeV   ({M1} + {M2} Msun binary)")
        show("M_dyn", ec.m_dyn, "Msun")
        show("v_dyn", ec.v_dyn, "c")
        show("M_disk", ec.m_disk, "Msun")
        show("<Ye>_dyn", ec.ye_dyn)
        show("f_tidal", ec.f_tidal)
        show("X_lanthanide", ec.x_lanthanide)
        fams = ", ".join(f"{k}={float(v):.3g}" for k, v in ec.m_dyn_by.items())
        print(f"  M_dyn by family: {fams}")

    g = jax.jacfwd(lambda L: ejecta_from_family(family_of(L), M1, M2)
                   .m_dyn.value)(60.0)
    print(f"\nAD gradient  d M_dyn / d L_sym = {float(g):.3e} Msun/MeV"
          "   (through metamodel + TOV + ejecta surrogate)")


if __name__ == "__main__":
    main()
