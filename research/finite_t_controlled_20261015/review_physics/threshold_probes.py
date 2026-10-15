"""Independent cold excitation-budget geometry for the local threshold model."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
from scipy.optimize import minimize,brentq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'urca_threshold'))
import threshold_study as study

checks=[];witnesses=[]
def add(name,error,tol):
 checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))

def best_polygon_gap(ref,E):
    # Allocate incident neutrino energy among an initial-neutron hole,
    # final-proton particle and final-electron particle, all nonnegative.
    # Momentum conservation is equivalent to the four lengths closing a polygon.
    M=study.eos.MASS;pn,pp=ref['pFn'],ref['pFp'];me=study.a.ME;mu=ref['mu_e_cold']
    def objective(x):
        dn,de=x;dp=E-dn-de
        lengths=np.array([math.sqrt(max(0.,pn*pn-2*M*dn)),E,
                          math.sqrt(max(0.,pp*pp+2*M*dp)),
                          math.sqrt((mu+de)**2-me*me)])
        return 2*max(lengths)-sum(lengths)
    bounds=[(0,min(E,ref['EF_n'])),(0,E)]
    constraints=[dict(type='ineq',fun=lambda x:E-x[0]-x[1])]
    trials=[]
    for seed in [[0,0],[min(E*.25,ref['EF_n']*.5),E*.25],[0,E*.75]]:
        opt=minimize(objective,seed,method='SLSQP',bounds=bounds,constraints=constraints,
                     options=dict(ftol=1e-12,maxiter=200))
        if not opt.success:raise ArithmeticError(str(opt.message))
        trials.append(opt)
    best=min(trials,key=lambda o:o.fun)
    return float(best.fun),best.x.tolist()

ref=study.cold_reference(.01,.1);M=study.eos.MASS
critical=brentq(lambda E:best_polygon_gap(ref,E)[0],.30,.32,xtol=1e-12)
pn,pp,pe=ref['pFn'],ref['pFp'],ref['pFe']
A=M+pn-pe;B=(pn-pe)**2-pp*pp
analytical=B/(A+math.sqrt(A*A-B))
add('independent energy-budget polygon root versus analytic onset',abs(critical/analytical-1),1e-9)
for E in [.09,.30,critical-1e-5,critical+1e-5,.9]:
    gap,alloc=best_polygon_gap(ref,E)
    expected=E>critical
    checks.append(dict(name=f'cold polygon support E={E}',gap_MeV=gap,passed=bool((gap<0)==expected)))
    witnesses.append(dict(E_nu_MeV=E,best_gap_MeV=gap,
                         neutron_hole_electron_excitation_MeV=alloc))
for Y in [1/9,.12]:
    r=study.cold_reference(.01,Y)
    gap,_=best_polygon_gap(r,.0001)
    checks.append(dict(name=f'above-triangle infinitesimal-E support Y={Y}',passed=bool(gap<0)))

report=json.loads((ROOT/'urca_threshold/qualified_results.json').read_text())
rows=report['rows']
sub=[x for x in rows if abs(x['reference']['Yp']-.1)<1e-12 and x['Enu']<critical]
for row in sub:
    checks.append(dict(name=f'qualified sub-onset positive log-chi T={row["T"]}',
                       log_chi=row['log_chi'],passed=bool(math.isfinite(row['log_chi']))))
offsets=[x['reference']['offset_neutron'] for x in rows if abs(x['reference']['Yp']-.1)<1e-12]
add('offset frozen across qualified temperatures',max(offsets)-min(offsets),1e-12)
paths=[ROOT/'urca_threshold/threshold_study.py',ROOT/'urca_threshold/cold_support.py']
out=dict(scope='Selected independent cold kinematic geometry, no complete Urca or EOS claim',
         checks=checks,all_passed=all(c['passed'] for c in checks),
         independent_onset_MeV=critical,analytical_onset_MeV=analytical,witnesses=witnesses,
         source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
Path(__file__).with_name('threshold_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
