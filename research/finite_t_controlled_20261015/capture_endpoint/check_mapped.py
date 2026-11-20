"""Independent-coordinate, endpoint, heavy-mass and KMS qualification."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import mapped_capture as m
import allowed_capture as a
ROOT=m.ROOT
sys.path.insert(0,str(ROOT/'crossed_opacity'))
sys.path.insert(0,str(ROOT/'allowed_opacity'))
from probe_frozen_capture import independent_q_energy
from check_allowed import classical_state
import static_capture as r


def main():
    checks=[]
    samples=[]
    def check(name,error,gate):
        checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(error<gate)))
    old=json.loads((ROOT/'crossed_opacity/frozen_capture_endpoint_probe.json').read_text())
    for c in old['cases']:
        st=a.OpticalState(a.PairState(**c['state']['baryons']),c['state']['mu_lepton'])
        E=c['Enu_MeV']
        logs=[m.log_coefficients(st,E,energy_order=n,angle_order=n) for n in (24,48,96)]
        vals=[math.exp(lc) for lj,lc in logs]
        ref=c['independent_q_energy_chi_s-1']
        tag=c['channel']+str(E)
        check('reference_'+tag,abs(vals[-1]/ref-1),3e-8)
        check('refinement_'+tag,abs(vals[-1]/vals[-2]-1),3e-8)
        check('KMS_'+tag,max(abs(lj-lc+(E-st.mu_neutrino)/st.baryons.T) for lj,lc in logs),3e-11)
        samples.append(dict(channel=c['channel'],E_MeV=E,chi_orders_24_48_96=vals,independent_chi=ref,
                            original_error24=c['relative_errors_against_independent'][1]))
    # Recompute the most sensitive threshold point in different q,Ee coordinates.
    pair=a.PairState(3,939,r.Q,0,r.Q+20,5)
    st=a.OpticalState(pair.reverse(),-5)
    ref=independent_q_energy(st,1.8)
    check('fresh_q_Ee_threshold',abs(math.exp(m.log_coefficients(st,1.8)[1])/ref-1),3e-8)
    static=r.State(3,5,8e-10,2e-10)
    for anti,channel in ((False,'electron_capture'),(True,'positron_capture')):
        st=classical_state(3,5,8e-10,2e-10,1e7,anti)
        for E in (3.,8.,20.):
            expected=r.spectral(static,E,channel)
            actual=np.exp(m.log_coefficients(st,E))
            check(f'heavy_static_{channel}_{E}',np.max(abs(actual/expected-1)),2e-5)
    for kw in ({'energy_order':4.5},{'angle_order':True},{'tail':0}):
        try:
            m.log_coefficients(st,3,**kw)
            ok=False
        except ValueError:
            ok=True
        check('invalid_control_'+str(kw),0 if ok else 1,.5)
    report=dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',checks=checks,samples=samples,
                fresh_threshold_reference=ref,scope='Numerical integration change only; frozen leading allowed current/NR phase space model',
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (Path(__file__),Path(m.__file__),Path(a.__file__),ROOT/'crossed_opacity/probe_frozen_capture.py')})
    Path(__file__).with_name('results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(checks),failed=[c for c in checks if not c['passed']]),indent=2))
    if report['status']!='PASS':
        raise SystemExit(1)

if __name__=='__main__':
    main()
