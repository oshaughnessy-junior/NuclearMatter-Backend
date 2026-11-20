"""Independent cold finite-energy support check for the local threshold toy."""
from pathlib import Path
import math,json
import numpy as np
from scipy.optimize import minimize_scalar
from threshold_study import cold_reference,eos,a

ref=cold_reference(.01,.1)
pn,pp,pe=ref['pFn'],ref['pFp'],ref['pFe'];M=eos.MASS
A=M+pn-pe;B=(pn-pe)**2-pp*pp
critical=B/(A+math.sqrt(A*A-B))

def maximum_cold_interval(Enu):
    # Energy conservation and occupied initial/final proton/electron states
    # restrict final electron Ee to [mu_e0,mu_e0+Enu]. Optimize the remaining
    # cold scalar-response interval, with q chosen to minimize its Emin.
    def interval(Ee):
        p=math.sqrt((Ee-a.ME)*(Ee+a.ME));qlo,qhi=abs(Enu-p),Enu+p
        w=Enu-Ee+ref['offset_neutron']
        stationary=math.sqrt(2*M*abs(w))
        q=min(qhi,max(qlo,stationary))
        emin=(M*w/q-q/2)**2/(2*M)
        return max(0.,ref['EF_n']-max(emin,ref['EF_p']-w))
    lo=ref['mu_e_cold'];hi=lo+Enu
    opt=minimize_scalar(lambda E:-interval(E),bounds=(lo,hi),method='bounded',options={'xatol':1e-13})
    return max(interval(lo),interval(hi),-float(opt.fun))

checks=[];samples=[]
for E in [.03,.09,.30,.32,.9]:
    value=maximum_cold_interval(E)
    expected_allowed=E>critical
    passed=(value>0)==expected_allowed
    checks.append(dict(name=f'cold finite-energy support E{E}',passed=bool(passed)))
    samples.append(dict(Enu=E,maximum_cold_interval_MeV=value,allowed=bool(value>0)))
out=dict(critical_Enu_MeV=critical,reference=ref,checks=checks,samples=samples,
         note='Selected local model finite-energy absorption onset; distinct from zero-neutrino cold Yp=1/9 triangle')
Path(__file__).with_name('cold_support_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
assert all(x['passed']for x in checks)
