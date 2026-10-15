"""Bounded numerical qualification; writes results.json beside this script."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit, gamma, zeta, log_expit

import static_capture as r


def relative(a, b):
    return float(np.max(np.abs(np.asarray(a)-b) / np.maximum(np.abs(b), 1e-250)))


def adaptive_moments(T, mu, channel):
    """Separate QUADPACK lepton-energy integral; shares constants/equations."""
    if channel == "electron_capture":
        lo, hi, sign = max(r.ME, r.Q), np.inf, 1
    elif channel == "positron_capture":
        lo, hi, sign = r.ME, np.inf, -1
    else:
        lo, hi, sign = r.ME, r.Q, 1

    # Strip a common exponential from suppressed integrals before QUADPACK's
    # absolute stopping rule; otherwise its relative error is meaningless.
    offset = min(0., (sign*mu-lo)/T) if channel != "neutron_decay" else min(0., (hi-mu)/T)
    def integrand(E, power):
        if channel == "electron_capture":
            nu = E-r.Q
        elif channel == "positron_capture":
            nu = E+r.Q
        else:
            nu = r.Q-E
        logblock = log_expit((sign*mu-E)/T) if channel != "neutron_decay" else log_expit((E-mu)/T)
        block = np.exp(logblock-offset)
        return E*np.sqrt(max(E*E-r.ME*r.ME, 0))*nu**power*block

    return np.exp(offset)*r.RATE_PREFACTOR * np.array([
        quad(lambda E: integrand(E, k), lo, hi, epsabs=1e-15, epsrel=2e-11, limit=300)[0]
        for k in (2, 3)
    ])


def main():
    checks = []
    samples = []
    def record(name, error, limit):
        checks.append({"name": name, "scaled_error": error, "gate": limit, "pass": bool(error < limit)})

    for T, mu in ((0.1, 0.), (1., -2.), (3., 5.), (10., 0.), (0.03, 8.)):
        for channel in ("electron_capture", "positron_capture", "neutron_decay"):
            vals = [r.target_moments(T, mu, channel, order=n) for n in (48, 96, 192, 384)]
            ref = adaptive_moments(T, mu, channel)
            record(f"adaptive_T{T}_mu{mu}_{channel}", relative(vals[-1], ref), 3e-7)
            record(f"refinement_T{T}_mu{mu}_{channel}", relative(vals[-1], vals[-2]), 1e-7)
            samples.append({"T_MeV": T, "mu_e_total_MeV": mu, "channel": channel,
                            "orders": [48, 96, 192, 384], "moments": [v.tolist() for v in vals],
                            "adaptive_moments": ref.tolist()})
    # Independent closed-form FD integrals: int x^k/(1+exp(x)) dx.
    for T in (0.5, 2., 8.):
        fd = lambda k: (1-2.**(-k))*gamma(k+1)*zeta(k+1, 1.)
        exact = r.RATE_PREFACTOR * np.array([T**5*fd(4), T**6*fd(5)])
        got = r.target_moments(T, 0., "electron_capture", 192, mass=0., q=0.)
        record(f"massless_Q0_FD_T{T}", relative(got, exact), 2e-12)
    # T->0, massless but finite Q: polynomial integral from Q to mu.
    mu, q = 8., r.Q
    exact = []
    for power in (2, 3):
        # E^2 (E-Q)^power, evaluated by polynomial antiderivative.
        poly = np.polynomial.Polynomial([-q, 1])**power * np.polynomial.Polynomial([0, 0, 1])
        integ = poly.integ()
        exact.append(r.RATE_PREFACTOR * (integ(mu)-integ(q)))
    got = r.target_moments(.001, mu, "electron_capture", 768, mass=0., q=q)
    record("degenerate_massless_Qfinite_limit", relative(got, np.array(exact)), 2e-5)
    # Restored absorption/emission equilibrium in common-mass classical targets.
    state = r.State(3., 5., 8e-8, 2e-8)
    mu_nu = state.mu_e-r.Q+state.T*np.log(state.n_p/state.n_n)
    for channel, energies in (("electron_capture", np.linspace(.01, 40, 200)),
                              ("positron_capture", np.linspace(r.Q+r.ME+1e-5, 40, 200)),
                              ("neutron_decay", np.linspace(.001, r.Q-r.ME-.001, 200))):
        j, chi = r.spectral(state, energies, channel)
        nu_mu = mu_nu if channel == "electron_capture" else -mu_nu
        ratio = np.exp(-(energies-nu_mu)/state.T)
        record(f"detailed_balance_{channel}", relative(j/chi, ratio), 1e-11)
        f = expit((nu_mu-energies)/state.T)
        residual = np.abs(j*(1-f)-chi*f) / np.maximum(j*(1-f)+chi*f, 1e-250)
        record(f"stationary_collision_{channel}", float(np.max(residual)), 1e-11)
    # Threshold tests, nonnegativity and spectrum moment normalization.
    j, chi = r.spectral(state, np.array([0, r.Q+r.ME-1e-8, r.Q+r.ME]), "positron_capture")
    record("positron_below_threshold", float(np.max(abs(j[:2]))), 1e-100)
    for channel in ("electron_capture", "positron_capture", "neutron_decay"):
        lo = r.Q+r.ME if channel == "positron_capture" else 0.
        hi = r.Q-r.ME if channel == "neutron_decay" else 200.
        density = state.n_p if channel == "electron_capture" else state.n_n
        moments = np.array([quad(lambda E: r.DOS_PREFACTOR*E**k*r.spectral(state, E, channel)[0], lo, hi,
                                 epsabs=1e-25, epsrel=2e-10, limit=200)[0] for k in (2, 3)])
        target = density*r.target_moments(state.T, state.mu_e, channel, 192)
        record(f"spectral_to_target_normalization_{channel}", relative(moments, target), 2e-9)
    vacuum_beta = r.target_moments(.01, -20., "neutron_decay", 192)[0]
    report = {
        "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "static classical free-nucleon ansatz; no dense-matter qualification",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "constants": {key: getattr(r, key) for key in ("HBAR", "HBARC", "GF", "VUD", "GA", "ME", "Q", "RATE_PREFACTOR")},
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(r.__file__))},
        "checks": checks, "convergence_samples": samples,
        "vacuum_neutron_beta_lifetime_seconds_uncorrected": float(1/vacuum_beta),
        "example_state": {"T_MeV": state.T, "mu_e_total_MeV": state.mu_e, "n_n_fm-3": state.n_n, "n_p_fm-3": state.n_p},
        "example_sources": r.source_terms(state, 192),
    }
    path = Path(__file__).with_name("results.json")
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "failed": [x for x in checks if not x["pass"]],
                      "uncorrected_beta_lifetime_s": 1/vacuum_beta, "results_path": str(path)}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
