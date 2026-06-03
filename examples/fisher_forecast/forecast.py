#!/usr/bin/env python3
"""Fisher forecast of EOS-parameter constraints from NS observables.

Demonstrates the differentiable backend's headline capability: propagate measurement
uncertainties on neutron-star observables (Lambda(1.4) from a GW event, R(1.4) from
NICER-like X-ray timing, M_max from a massive pulsar) into forecast constraints on
nuclear EOS parameters, using forward-mode sensitivities (dObs/dtheta) assembled by
the implicit-function / envelope theorems (see nmbackend.inference.fisher).

This is a *forecast* (Gaussian/linearized), not a full posterior -- but it is exact in
the sensitivities and needs no sampling, so it is a fast way to see which parameters a
given measurement set constrains and how they correlate.

Run:  PYTHONPATH=src python examples/fisher_forecast/forecast.py
"""
import numpy as np

from nmbackend.eos import MetaModelEOS
from nmbackend.inference import sensitivity_matrix, eos_observables, fisher_covariance

# --- fiducial EOS and the parameters we want to constrain ------------------------
mm = MetaModelEOS()
fiducial = {"L_sym": 60.0, "K_sym": -100.0, "cs2_tail": 0.5}
param_names = ["L_sym", "K_sym", "cs2_tail"]
param_units = {"L_sym": "MeV", "K_sym": "MeV", "cs2_tail": "c^2"}

# --- the measurement set: observable, 1-sigma uncertainty ------------------------
#   Lambda(1.4): GW170817-class tidal measurement
#   R(1.4):      NICER-class radius (~5%)
#   M_max:       massive-pulsar lower bound treated as a soft measurement
obs_kw = dict(masses_lambda=(1.4,), masses_radius=(1.4,), include_mmax=True,
              hc_grid_n=160)
sigma_frac = {"Lambda(1.4)": 0.30, "R(1.4)": 0.05, "M_max": 0.05}


def main():
    vals, names = eos_observables(mm, fiducial, **obs_kw)
    vals = np.asarray(vals)
    sigma = np.array([sigma_frac[n] * abs(v) for n, v in zip(names, vals)])

    J, obs_names, pnames = sensitivity_matrix(mm, fiducial, param_names, **obs_kw)
    J = np.asarray(J)

    cov = np.asarray(fisher_covariance(J, sigma))
    err = np.sqrt(np.diag(cov))
    corr = cov / np.outer(err, err)

    print("Fiducial EOS:", {k: fiducial[k] for k in param_names})
    print("\nObservables (fiducial value +/- assumed 1-sigma):")
    for n, v, s in zip(obs_names, vals, sigma):
        print(f"  {n:12s} = {v:10.3f}  +/- {s:8.3f}  ({s/abs(v)*100:.0f}%)")

    print("\nSensitivity matrix  dObs/dtheta:")
    hdr = "  " + " ".join(f"{p:>10s}" for p in pnames)
    print(hdr)
    for i, n in enumerate(obs_names):
        print(f"  {n:12s}" + " ".join(f"{J[i, j]:10.3f}" for j in range(len(pnames))))

    print("\nForecast 1-sigma parameter constraints (this measurement set):")
    for p, e in zip(pnames, err):
        print(f"  {p:9s} = {fiducial[p]:8.2f} +/- {e:7.2f} {param_units[p]}")

    print("\nForecast correlation matrix:")
    for i, p in enumerate(pnames):
        print(f"  {p:9s} " + " ".join(f"{corr[i, j]:+.2f}" for j in range(len(pnames))))


if __name__ == "__main__":
    main()
