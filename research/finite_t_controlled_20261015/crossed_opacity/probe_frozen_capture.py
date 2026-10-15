"""Read-only endpoint probe of frozen allowed_capture, with exact records.

The independent q/Ee reference is integrated with tight absolute tolerance;
the frozen optical module is not changed by this script.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

import crossed_decay as d
import allowed_capture as a
from free_density_response import response
import static_capture as r


def independent_q_energy(state,Enu):
    s=state.baryons
    def integrand(Ee):
        pe=math.sqrt(max(0.,Ee*Ee-r.ME*r.ME))
        if pe==0:
            return 0.
        def f(q):
            c=(Enu*Enu+pe*pe-q*q)/(2*Enu*pe)
            return Ee*q/Enu*a.angular_factor(Ee,c)*response(s,q,Enu-Ee)
        v=quad(f,abs(Enu-pe),Enu+pe,epsabs=1e-40,epsrel=3e-10,limit=250)[0]
        return v*expit((Ee-state.mu_lepton)/s.T)
    upper=max(r.ME,Enu-(s.offset_b-s.offset_a),state.mu_lepton)+60*s.T+30
    edges=np.linspace(r.ME,upper,301)
    v=sum(quad(integrand,lo,hi,epsabs=1e-40,epsrel=5e-10,limit=200)[0]
          for lo,hi in zip(edges[:-1],edges[1:]))
    return a.K*v


def main():
    pair=d.PairState(3,939,r.Q,0,r.Q+20,5)
    cases=[]
    for anti,energies in ((False,(.1,1.,5.)),(True,(1.,1.5,1.8,2.,3.))):
        state=a.OpticalState(pair.reverse() if anti else pair,-5. if anti else 5.)
        for Enu in energies:
            orders=(12,24,48,96,192)
            vals=[math.exp(a.log_coefficients(state,Enu,energy_order=n,angle_order=n)[1]) for n in orders]
            ref=independent_q_energy(state,Enu)
            cases.append({"channel":"antinu_p_to_n" if anti else "nu_n_to_p","state":asdict(state),
                          "Enu_MeV":Enu,"orders_energy_and_angle":orders,"chi_s-1":vals,
                          "independent_q_energy_chi_s-1":ref,
                          "relative_errors_against_independent":[v/ref-1 for v in vals],
                          "relative_24_to_48":vals[1]/vals[2]-1,
                          "relative_96_to_192":vals[3]/vals[4]-1})
    out={"status":"DIAGNOSTIC_ONLY","created_utc":datetime.now(timezone.utc).isoformat(),
         "purpose":"Read-only extension probes outside previously selected capture qualification states",
         "frozen_capture_sha256":hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
         "probe_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"cases":cases}
    Path(__file__).with_name("frozen_capture_endpoint_probe.json").write_text(json.dumps(out,indent=2)+"\n")
    for c in cases:
        print(c["channel"],c["Enu_MeV"],"relerr24",c["relative_errors_against_independent"][1],
              "relerr192",c["relative_errors_against_independent"][-1])


if __name__=="__main__":
    main()
