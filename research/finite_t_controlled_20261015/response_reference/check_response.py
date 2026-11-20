"""Bounded independent phase-space and limiting tests for density response."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit, log_expit
from numpy.polynomial.legendre import leggauss

import free_density_response as r


def transverse_phase_space(s, q, omega):
    """Integrate actual FD phase space in p_perp after exact p_z delta.

    Shares normalization and dispersion, not the analytic log formula.
    """
    pz = s.mass*(omega-s.offset_b+s.offset_a)/q-q/2
    E0 = pz*pz/(2*s.mass)
    w = omega-s.offset_b+s.offset_a
    peak_E = max(E0, (s.kinetic_mu_a+s.kinetic_mu_b-w)/2)
    def logproduct(E):
        return log_expit((s.kinetic_mu_a-E)/s.T)+log_expit((E+w-s.kinetic_mu_b)/s.T)
    scale = logproduct(peak_E)
    def f(y):
        E = E0+s.T*y*y
        return y*np.exp(logproduct(E)-scale)
    # Explicit panels resolve both Fermi edges in transverse momentum.
    endpoints = sorted(set([0.] + [math.sqrt((E-E0)/s.T) for E in
                                  (s.kinetic_mu_a, s.kinetic_mu_b-w, peak_E)
                                  if E>E0]))+[np.inf]
    val, err = 0., 0.
    for a, b in zip(endpoints[:-1], endpoints[1:]):
        v, e = quad(f, a, b, epsabs=2e-12, epsrel=2e-12, limit=250)
        val += v
        err += e
    return math.log(s.mass/(math.pi*q*r.HBARC**3)*2*s.mass*s.T)+scale+math.log(val), err/val


def density(s):
    # spin2 phase space, unrelated to dynamical response analytical reduction
    value = quad(lambda p: p*p*expit((s.kinetic_mu_a-p*p/(2*s.mass))/s.T),
                 0, np.inf, epsabs=1e-8, epsrel=2e-11, limit=250)[0]
    return value/(np.pi**2*r.HBARC**3)


def direct_static_sum(s, q, order=64):
    """Independent p,cos(theta) quadrature of f_a(1-f_b), no energy delta."""
    cs, ws = leggauss(order)
    def integrand(p):
        energy_a = p*p/(2*s.mass)
        energy_b = (p*p+q*q+2*p*q*cs)/(2*s.mass)
        return p*p*expit((s.kinetic_mu_a-energy_a)/s.T)*np.dot(ws, expit((energy_b-s.kinetic_mu_b)/s.T))
    val = quad(integrand, 0, np.inf, epsabs=1e-9, epsrel=1e-10, limit=250)[0]
    return val/(2*np.pi**2*r.HBARC**3)


def main():
    checks, samples = [], []
    def add(name, err, gate):
        checks.append({"name": name, "error": float(err), "gate": gate, "pass": bool(err<gate)})
    states = [r.PairState(3, 939, 939.565, 938.272, 969.565, 943.272),
              r.PairState(8, 800, 700, 715, 684, 683),
              r.PairState(.1, 939, 939.565, 938.272, 969.565, 943.272),
              r.PairState(1, 939, 0, 10, -700, -680)]
    for i,s in enumerate(states):
        for q,omega in ((10,0),(40,-8),(100,12),(75,s.mu_b-s.mu_a),
                        (75,s.mu_b-s.mu_a+1e-8*s.T),(75,s.mu_b-s.mu_a-1e-8*s.T)):
            analytic = r.log_response(s,q,omega)
            numerical, estimate = transverse_phase_space(s,q,omega)
            add(f"phase_space_state{i}_q{q}_w{omega}", abs(analytic-numerical), 2e-9)
            z = (omega+s.mu_a-s.mu_b)/s.T
            reverse = r.log_response(s.reverse(),q,-omega)
            add(f"KMS_state{i}_q{q}_w{omega}", abs(reverse-analytic+z), 3e-9)
            samples.append({"state": asdict(s),"q_MeV":q,"omega_MeV":omega,
                            "log_S":analytic,"log_S_independent":numerical,
                            "independent_relative_error_estimate":estimate})
    # Classical MB response: final blocking negligible, initial fugacity tiny.
    dilute = r.PairState(3,939,0,5,-60,-70)
    for q,omega in ((40,0),(80,7),(120,-4)):
        emin = r.kinematic_lower_energy(dilute,q,omega)
        expected = (2*math.log(dilute.mass)+math.log(dilute.T)-math.log(math.pi*q)
                    -3*math.log(r.HBARC)+(dilute.kinetic_mu_a-emin)/dilute.T)
        add(f"MB_limit_q{q}_w{omega}",abs(r.log_response(dilute,q,omega)-expected),1e-8)
    cold = r.PairState(.001,939,0,2,30,7)
    for q,omega in ((100,5),(200,8),(150,1)):
        exact = r.cold_response(cold.mass,cold.offset_a,cold.offset_b,cold.mu_a,cold.mu_b,q,omega)
        got = r.response(cold,q,omega)
        add(f"cold_allowed_q{q}_w{omega}",abs(got/exact-1),2e-10)
    for omega in (-100,100):
        exact = r.cold_response(cold.mass,cold.offset_a,cold.offset_b,cold.mu_a,cold.mu_b,100,omega)
        add(f"cold_forbidden_w{omega}",abs(exact),1e-100)
        # finite-T response is exponentially small but not kinematically zero
        add(f"finite_T_tail_w{omega}",0. if math.isfinite(r.log_response(cold,100,omega)) else 1.,.5)
    # Static sum and exact neutral-current f sum: spin2 and 2pi delta checked.
    neutral = r.PairState(3,939,0,0,30,30)
    for q in (30,100):
        s0 = quad(lambda w:r.response(neutral,q,w)/(2*np.pi),-200,200,epsabs=1e-12,epsrel=2e-10,limit=250)[0]
        direct = direct_static_sum(neutral,q)
        add(f"static_sum_q{q}",abs(s0/direct-1),2e-9)
        fsum = quad(lambda w:w*r.response(neutral,q,w)/(2*np.pi),-200,200,epsabs=1e-12,epsrel=2e-10,limit=250)[0]
        expected = density(neutral)*q*q/(2*neutral.mass)
        add(f"neutral_f_sum_q{q}",abs(fsum/expected-1),2e-9)
    # Unequal composition and offsets: independent equal-time CC sum rule.
    for i in (0,1):
        s=states[i]
        q=70.
        value=quad(lambda w:r.response(s,q,w)/(2*np.pi),-400,400,epsabs=1e-12,epsrel=2e-10,limit=250)[0]
        direct=direct_static_sum(s,q,96)
        add(f"CC_static_sum_state{i}",abs(value/direct-1),2e-9)
        add(f"CC_angular_refinement_state{i}",abs(direct/direct_static_sum(s,q,48)-1),2e-9)
    # A shared energy-zero shift changes no response or chemical imbalance.
    original=states[0]
    shifted=r.PairState(original.T,original.mass,original.offset_a+123.,original.offset_b+123.,
                        original.mu_a+123.,original.mu_b+123.)
    add("common_energy_zero_invariance",abs(r.log_response(original,70,4)-r.log_response(shifted,70,4)),2e-12)
    # q=0 is a delta in energy, not a scalar floating point value.
    for badq,badw in ((0,0),(-1,0),(np.inf,0),(1,np.nan)):
        try:
            r.response(neutral,badq,badw)
            guarded = False
        except ValueError:
            guarded = True
        add(f"invalid_transfer_q{badq}_w{badw}",0 if guarded else 1,.5)
    report={"status":"PASS" if all(x["pass"] for x in checks) else "FAIL",
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "model":"common-mass NR spin2 scalar density-transition response; not physical full weak tensor",
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
            "sources_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(r.__file__))},
            "checks":checks,"phase_space_samples":samples}
    out=Path(__file__).with_name("results.json")
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"checks":len(checks),"failed":[x for x in checks if not x["pass"]]},indent=2))
    if report["status"]!="PASS":
        raise SystemExit(1)


if __name__=="__main__":
    main()
