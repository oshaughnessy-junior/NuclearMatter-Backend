"""Absolute normalization, heavy/static limit and quadrature qualification."""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import expit

import allowed_capture as a
from free_density_response import response
import static_capture as r


def classical_state(T,mu_e,n_n,n_p,mass,anti=False):
    phase=2*(mass*T/(2*math.pi))**1.5/r.HBARC**3
    nnmu=r.Q+T*math.log(n_n/phase)
    npmu=T*math.log(n_p/phase)
    ns=a.PairState(T,mass,r.Q,0,nnmu,npmu)
    return a.OpticalState(ns.reverse() if anti else ns,-mu_e if anti else mu_e)


def q_energy_adaptive(state,Enu):
    """Different integration variables q,Ee and QUADPACK, same theory/S."""
    s=state.baryons
    def energy_integrand(Ee):
        pe=math.sqrt(max(0.,Ee*Ee-r.ME*r.ME))
        if pe==0:
            return 0.
        qlo,qhi=abs(Enu-pe),Enu+pe
        def q_integrand(q):
            c=(Enu*Enu+pe*pe-q*q)/(2*Enu*pe)
            return Ee*q/Enu*a.angular_factor(Ee,c)*response(s,q,Enu-Ee)
        value=quad(q_integrand,qlo,qhi,epsabs=1e-12,epsrel=2e-9,limit=200)[0]
        return value*expit((Ee-state.mu_lepton)/s.T)
    # use an independent coarse grid as panels, not the production panel helper
    upper=max(r.ME,Enu-(s.offset_b-s.offset_a),state.mu_lepton)+50*s.T+100
    edges=np.linspace(r.ME,upper,121)
    value=sum(quad(energy_integrand,lo,hi,epsabs=1e-11,epsrel=3e-8,limit=100)[0]
              for lo,hi in zip(edges[:-1],edges[1:]))
    return a.K*value


def spectral_moments(state,order=32,cutoff=160):
    """Integrate neutrino emission with extra threshold and thermal panels."""
    # Separate each side of vacuum positron threshold, with smooth sqrt maps.
    threshold=r.Q+r.ME if state.mu_lepton<0 else 0.
    edges=sorted(set([0.,threshold,5.,15.,40.,80.,float(cutoff)]))
    xs,ws=a.nodes(order)
    result=np.zeros(2)
    for lo,hi in zip(edges[:-1],edges[1:]):
        # A square-root map smooths the static-limit upper-threshold corner.
        xx=.5*(xs+1)*math.sqrt(hi-lo)
        energies=lo+xx*xx
        weights=.5*math.sqrt(hi-lo)*ws*2*xx
        for E,w in zip(energies,weights):
            lj,lc=a.log_coefficients(state,float(E),energy_order=24,angle_order=24)
            j=math.exp(lj)
            result+=r.DOS_PREFACTOR*w*np.array([E*E*j,E**3*j])
    return result


def main():
    checks,static_samples,moment_samples=[],[],[]
    def add(name,error,gate):
        checks.append({"name":name,"error":float(error),"gate":gate,"pass":bool(error<gate)})
    T,mu,nn,np_=3.,5.,8e-10,2e-10
    static=r.State(T,mu,nn,np_)
    for anti,channel in ((False,"electron_capture"),(True,"positron_capture")):
        for Enu in (3.,8.,20.):
            sj,sc=r.spectral(static,Enu,channel)
            for mass in (939.,1e4,1e5,1e7):
                st=classical_state(T,mu,nn,np_,mass,anti)
                lj,lc=a.log_coefficients(st,Enu)
                gj,gc=math.exp(lj),math.exp(lc)
                sample={"channel":channel,"E_nu_MeV":Enu,"kinetic_mass_MeV":mass,
                        "j_s-1":gj,"chi_s-1":gc,"static_j_s-1":float(sj),"static_chi_s-1":float(sc),
                        "chi_static_relative_difference":float(gc/sc-1)}
                static_samples.append(sample)
                add(f"KMS_{channel}_E{Enu}_M{mass}",abs(lj-lc+(Enu-st.mu_neutrino)/T),2e-11)
                if mass==1e7:
                    add(f"absolute_static_limit_{channel}_E{Enu}",max(abs(gj/sj-1),abs(gc/sc-1)),2e-5)
                    Ee=Enu+(r.Q if not anti else -r.Q)
                    pe=math.sqrt(Ee*Ee-r.ME*r.ME)
                    # Independent bare vacuum cross section in cm^2, with the
                    # specified final charged-lepton vacancy included.
                    sigma=r.GF**2*r.VUD**2*r.HBARC**2/math.pi*(1+3*r.GA*r.GA)*Ee*pe*1e-26
                    vacancy=expit((Ee-(-mu if anti else mu))/T)
                    target_density=np_ if anti else nn
                    from_cross_section=target_density*1e39*a.C_CM_S*sigma*vacancy
                    add(f"absolute_cross_section_cm2_{channel}_E{Enu}",abs(gc/from_cross_section-1),2e-5)
    # Independent finite-mass/degenerate numerical coordinates.
    degenerate=a.OpticalState(a.PairState(3,939,r.Q,0,r.Q+20,5),5)
    for Enu in (5.,20.):
        vals=[a.coefficients(degenerate,Enu,energy_order=n,angle_order=n)["chi_s-1"] for n in (12,24,48)]
        independent=q_energy_adaptive(degenerate,Enu)
        add(f"quadrature_refinement_E{Enu}",abs(vals[-1]/vals[-2]-1),3e-6)
        add(f"q_energy_independent_E{Enu}",abs(vals[-1]/independent-1),3e-6)
        base=a.coefficients(degenerate,Enu,energy_order=32,angle_order=32,tail=40)
        extended=a.coefficients(degenerate,Enu,energy_order=32,angle_order=32,tail=70)
        add(f"energy_tail_E{Enu}",abs(base["chi_s-1"]/extended["chi_s-1"]-1),1e-7)
        add(f"unit_conversion_E{Enu}",abs(base["true_absorption_cm-1"]*a.C_CM_S/base["chi_s-1"]-1),1e-14)
    # Absolute emitted number/energy moments, with no neutron beta component.
    for anti,channel,target in ((False,"electron_capture",np_),(True,"positron_capture",nn)):
        state=classical_state(T,mu,nn,np_,1e7,anti)
        vals=[spectral_moments(state,order=n) for n in (16,32)]
        expected=target*r.target_moments(T,mu,channel,192)
        err=float(np.max(abs(vals[-1]/expected-1)))
        add(f"absolute_number_energy_moments_{channel}",err,3e-5)
        add(f"outer_neutrino_refinement_{channel}",float(np.max(abs(vals[-1]/vals[-2]-1))),3e-5)
        moment_samples.append({"channel":channel,"orders":[16,32],"number_energy_moments":[v.tolist() for v in vals],
                               "static_number_energy_moments":expected.tolist()})
    # Angular mass factor has correct threshold and angular integral.
    for Ee in (r.ME,1.,10.):
        val=quad(lambda c:a.angular_factor(Ee,c),-1,1)[0]
        add(f"angular_integral_Ee{Ee}",abs(val/(2*(a.GV*a.GV+3*a.GA*a.GA))-1),1e-14)
    report={"status":"PASS" if all(x["pass"] for x in checks) else "FAIL",
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "model":"allowed NR V0/Aspin current with common-mass free-FD recoil/Pauli response; captures only",
            "prefactor_K":a.K,"degenerate_test_state":asdict(degenerate),
            "constants":{key:getattr(r,key) for key in ("GF","VUD","GA","HBAR","HBARC","ME","Q")},
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
            "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                             (Path(__file__),Path(a.__file__),Path(a.ROOT/"response_reference/free_density_response.py"),
                              Path(r.__file__))},
            "checks":checks,"static_limit_samples":static_samples,"moment_samples":moment_samples}
    Path(__file__).with_name("results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"checks":len(checks),"failed":[x for x in checks if not x["pass"]]},indent=2))
    if report["status"]!="PASS":
        raise SystemExit(1)


if __name__=="__main__":
    main()
