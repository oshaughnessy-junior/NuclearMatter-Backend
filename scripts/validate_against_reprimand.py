#!/usr/bin/env python3
"""Cross-validate the nmbackend TOV solver against RePrimAnd (and the in-process
SciPy reference).  Run this in the validation environment (docs/environment.md):

    PYTHONPATH=src python scripts/validate_against_reprimand.py

Prints M, R, Lambda from three independent solvers on a grid of central densities
and the max relative differences.  RePrimAnd is the external high-accuracy truth;
the SciPy radius-domain solver is an independent in-process check.  If RePrimAnd is
not installed, only the diffrax-vs-SciPy comparison runs.
"""
import numpy as np

from nmbackend.eos import MetaModelEOS
from nmbackend.tov.solver import solve_structure
from nmbackend.reference.scipy_tov import solve_reference
from nmbackend.reference import reprimand as rpa


def main():
    eos = MetaModelEOS()
    table = eos.eos_table()
    fracs = np.array([0.10, 0.20, 0.35, 0.55, 0.75])
    hcs = fracs * float(table.h[-1])

    print(f"{'h_c':>8} {'M_diffrax':>10} {'M_scipy':>10} {'M_repr':>10} "
          f"{'R_diffrax':>10} {'R_scipy':>10} {'R_repr':>10} "
          f"{'L_diffrax':>10} {'L_scipy':>10} {'L_repr':>10}")

    # RePrimAnd sequence (interpolated to our central masses for comparison)
    repr_ok = rpa.HAVE_REPRIMAND
    if repr_ok:
        try:
            Mr, Rr, Lr, _ = rpa.reprimand_sequence(table)
        except Exception as e:  # API drift etc. -- report, don't crash
            print(f"\n[RePrimAnd present but call failed: {e!r}]")
            print("  -> adjust the lines marked '# VERIFY' in reference/reprimand.py")
            repr_ok = False

    # Solve the whole grid first so we can locate our own maximum-mass turnover.
    # RePrimAnd's make_tov_branch_stable returns ONLY the stable branch (M rising
    # to M_max), and the comparison below matches at gravitational mass.  Past
    # M_max our sequence is on the unstable branch where M decreases, so a
    # matched-mass lookup lands on RePrimAnd's low-density rising branch -- a
    # different star entirely (NOT a solver disagreement).  Flag those rows.
    rows = [(hc,
             solve_structure(table, hc, backend="diffrax", rtol=1e-8, atol=1e-10),
             solve_reference(table, hc, rtol=1e-9, atol=1e-11))
            for hc in hcs]
    masses = [float(a.M_msun) for _, a, _ in rows]
    i_max = int(np.argmax(masses))  # last stable point; rows after this are unstable

    dM = dR = dL = 0.0
    for i, (hc, a, b) in enumerate(rows):
        unstable = i > i_max
        if repr_ok and not unstable:
            Mc = float(a.M_msun)
            mr = np.interp(Mc, Mr, Rr); ml = np.interp(Mc, Mr, Lr)
            r_repr, l_repr = f"{mr:10.3f}", f"{ml:10.2f}"
        elif repr_ok:  # past M_max -- matched-mass comparison is ill-defined
            r_repr = l_repr = f"{'unstable':>10}"
        else:
            r_repr = l_repr = f"{'nan':>10}"
        print(f"{hc:8.4f} {float(a.M_msun):10.4f} {b.M_msun:10.4f} {'-':>10} "
              f"{float(a.R_km):10.3f} {b.R_km:10.3f} {r_repr} "
              f"{float(a.Lambda):10.2f} {b.Lambda:10.2f} {l_repr}")
        dM = max(dM, abs(float(a.M_msun) - b.M_msun) / b.M_msun)
        dR = max(dR, abs(float(a.R_km) - b.R_km) / b.R_km)
        dL = max(dL, abs(float(a.Lambda) - b.Lambda) / abs(b.Lambda))

    print(f"\ndiffrax vs SciPy reference  max rel diff: "
          f"M {dM:.2e}  R {dR:.2e}  Lambda {dL:.2e}")
    if repr_ok and i_max < len(rows) - 1:
        print(f"({len(rows) - 1 - i_max} grid point(s) past M_max="
              f"{masses[i_max]:.3f} Msun are on the unstable branch; RePrimAnd "
              f"matched-mass comparison skipped there.)")
    if not repr_ok:
        print("RePrimAnd not available -- install pyreprimand (docs/environment.md) "
              "to add the external cross-check.")


if __name__ == "__main__":
    main()
