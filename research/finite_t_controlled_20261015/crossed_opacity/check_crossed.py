"""Crossing signs, heavy/static normalization and thermal support checks."""
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
from scipy.special import expit

import crossed_decay as d
import allowed_capture as a
from free_density_response import response
import static_capture as r


def classical_state(T,mu,nn,np_,M):
    thermal=2*(M*T/(2*math.pi))**1.5/r.HBARC**3
    s=d.PairState(T,M,r.Q,0,r.Q+T*math.log(nn/thermal),T*math.log(np_/thermal))
    return d.DecayState(s,mu)


def q_energy_adaptive(state,Enu,absolute_tolerance=1e-24):
    """Independent integration in q,Ee; no shared optical panel generator."""
    s=state.baryons
    def integrand(Ee):
        pe=math.sqrt(max(0.,Ee*Ee-r.ME*r.ME))
        if pe==0:
            return 0.
        def f(q):
            c=(q*q-Enu*Enu-pe*pe)/(2*Enu*pe)
            return Ee*q/Enu*a.angular_factor(Ee,c)*response(s,q,-Enu-Ee)
        val=quad(f,abs(Enu-pe),Enu+pe,epsabs=absolute_tolerance,epsrel=2e-10,limit=250)[0]
        return val*expit((Ee-state.mu_e)/s.T)
    upper=max(r.ME,r.Q-Enu,state.mu_e)+60*s.T+30
    edges=np.linspace(r.ME,upper,201)
    val=sum(quad(integrand,lo,hi,epsabs=absolute_tolerance,epsrel=3e-10,limit=150)[0]
            for lo,hi in zip(edges[:-1],edges[1:]))
    return a.K*val


def decay_moments(state,order=20,cutoff=1.4):
    """Number and antinu-energy per volume, with thermal endpoint panels."""
    threshold=r.Q-r.ME
    width=r.Q*math.sqrt(state.baryons.T/state.baryons.mass)
    edges=sorted(set([0.,.2,.5,threshold,float(cutoff)]+[
        threshold+k*width for k in (-16,-8,-4,-1,1,4,8,16)
        if 0<threshold+k*width<cutoff]))
    xs,ws=a.nodes(order)
    ans=np.zeros(2)
    for lo,hi in zip(edges[:-1],edges[1:]):
        xx=.5*(xs+1)*math.sqrt(hi-lo)
        Es=hi-xx*xx if hi<=threshold else lo+xx*xx
        weights=.5*math.sqrt(hi-lo)*ws*2*xx
        for E,w in zip(Es,weights):
            lj,lc=d.log_coefficients(state,float(E),energy_order=20,angle_order=20)
            j=math.exp(lj)
            ans+=r.DOS_PREFACTOR*w*np.array([E*E*j,E**3*j])
    return ans


def main():
    checks,samples,moments=[],[],[]
    def add(name,error,gate):
        checks.append({"name":name,"error":float(error),"gate":gate,"pass":bool(error<gate)})
    T,mu,nn,np_=3.,5.,8e-10,2e-10
    for M in (939.,1e4,1e5,1e7):
        state=classical_state(T,mu,nn,np_,M)
        for Enu in (.1,.4,.7):
            lj,lc=d.log_coefficients(state,Enu)
            j,chi=math.exp(lj),math.exp(lc)
            sj,sc=r.spectral(r.State(T,mu,nn,np_),Enu,"neutron_decay")
            add(f"KMS_E{Enu}_M{M}",abs(lj-lc+(Enu-state.mu_antineutrino)/T),2e-11)
            if M==1e7:
                add(f"absolute_static_j_chi_E{Enu}",max(abs(j/sj-1),abs(chi/sc-1)),5e-6)
            samples.append({"mass_MeV":M,"E_nu_MeV":Enu,"j_s-1":j,"chi_s-1":chi,
                            "static_j_s-1":float(sj),"static_chi_s-1":float(sc)})
    finite=classical_state(T,mu,nn,np_,939.)
    for E in (.4,1.):
        vals=[d.coefficients(finite,E,energy_order=n,angle_order=n)["j_s-1"] for n in (12,24,48)]
        other=q_energy_adaptive(finite,E)
        add(f"quadrature_refinement_E{E}",abs(vals[-1]/vals[-2]-1),2e-6)
        add(f"q_energy_independent_E{E}",abs(vals[-1]/other-1),2e-6)
        extended=d.coefficients(finite,E,energy_order=32,angle_order=32,tail=70)["j_s-1"]
        add(f"tail_extension_E{E}",abs(vals[-1]/extended-1),2e-6)
        if E==1.:
            plateau=q_energy_adaptive(finite,E,1e-20)
            add("independent_tail_tolerance_plateau",abs(plateau/other-1),2e-7)
    degenerate=d.DecayState(d.PairState(3,939,r.Q,0,r.Q+20,5),5)
    for E in (.4,1.):
        lj,lc=d.log_coefficients(degenerate,E,energy_order=48,angle_order=48)
        coarsej,coarsec=d.log_coefficients(degenerate,E,energy_order=24,angle_order=24)
        independent=q_energy_adaptive(degenerate,E)
        add(f"degenerate_KMS_E{E}",abs(lj-lc+(E-degenerate.mu_antineutrino)/3),2e-11)
        add(f"degenerate_refinement_E{E}",max(abs(math.expm1(lj-coarsej)),abs(math.expm1(lc-coarsec))),2e-6)
        add(f"degenerate_independent_E{E}",abs(math.exp(lj)/independent-1),2e-6)
    # Former static antinu gap: thermal NR phase space fills it mathematically.
    for E in (.9,1.,1.2):
        lj,lc=d.log_coefficients(finite,E,energy_order=32,angle_order=32)
        staticj=r.spectral(r.State(T,mu,nn,np_),E,"neutron_decay")[0]
        add(f"cold_forbidden_finite_thermal_E{E}",0 if staticj==0 and math.isfinite(lj) else 1,.5)
        combo=d.combined_antineutrino(finite,E,energy_order=24,angle_order=24)
        add(f"combined_antinu_KMS_E{E}",abs(combo["log_j_s-1"]-combo["log_chi_s-1"]+(E-finite.mu_antineutrino)/T),2e-10)
        samples.append({"mass_MeV":939.,"E_nu_MeV":E,"thermal_log_j_s-1":lj,
                        "static_neutron_decay_j_s-1":float(staticj),"combined_antinu":combo})
    # Warm moments and dilute-vacuum uncorrected lifetime, same heavy-mass limit.
    for label,t,em in (("warm",3.,5.),("vacuum_leptons",.01,-20.)):
        state=classical_state(t,em,nn,np_,1e7)
        vals=[decay_moments(state,n) for n in (12,24)]
        expected=nn*r.target_moments(t,em,"neutron_decay",192)
        add(f"heavy_mass_number_energy_moments_{label}",float(np.max(abs(vals[-1]/expected-1))),1e-5)
        add(f"outer_nu_refinement_{label}",float(np.max(abs(vals[-1]/vals[-2]-1))),2e-6)
        moment={"label":label,"T_MeV":t,"mu_e_MeV":em,"orders":[12,24],
                "moments":[v.tolist() for v in vals],"static_moments":expected.tolist()}
        if label=="vacuum_leptons":
            moment["uncorrected_beta_lifetime_s"]=nn/vals[-1][0]
            moment["static_uncorrected_beta_lifetime_s"]=nn/expected[0]
        moments.append(moment)
    # Pure angular correlation integral validates a_beta, not just total W.
    Ee=1.
    beta=math.sqrt(1-r.ME*r.ME/Ee**2)
    norm=quad(lambda c:a.angular_factor(Ee,c),-1,1)[0]
    mean=quad(lambda c:c*a.angular_factor(Ee,c),-1,1)[0]/norm
    exact=beta*(a.GV*a.GV-a.GA*a.GA)/(3*(a.GV*a.GV+3*a.GA*a.GA))
    add("electron_antinu_angular_correlation",abs(mean-exact),2e-14)
    out={"status":"PASS" if all(x["pass"] for x in checks) else "FAIL",
         "created_utc":datetime.now(timezone.utc).isoformat(),
         "model":"crossed n->p electron+antinu LOallowedcurrent; common-mass NR dynamical phase space",
         "constants":{key:getattr(r,key) for key in ("GF","VUD","GA","HBAR","HBARC","ME","Q")},
         "finite_test_state":asdict(finite),"prefactor_K":a.K,
         "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__},
         "sources_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (Path(__file__),Path(d.__file__),Path(a.__file__),Path(r.__file__),
                             Path(d.ROOT/"response_reference/free_density_response.py"))},
         "checks":checks,"spectral_samples":samples,"moment_samples":moments}
    Path(__file__).with_name("results.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":out["status"],"checks":len(checks),"failed":[x for x in checks if not x["pass"]]},indent=2))
    if out["status"]!="PASS":
        raise SystemExit(1)


if __name__=="__main__":
    main()
