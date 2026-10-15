"""Independent isentropic-parameter-work probe of frozen RMF reference."""
from pathlib import Path
import sys,json,math
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rmf_parameter_ad'))
import gm1_parameter_ad as ad

n,T,Y=.16,10.,.1
x=ad.pack_state(n,T,Y)
y,J,diag=ad.oracle_value_jacobian(x,return_diagnostics=True)
entropy=y[11]
checks=[]
def add(name,error,tol):
    checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))

# Adjust temperature by a separate scalar entropy root, not the AD entropy row.
for key,column in [('C_sigma',3),('C_omega',4),('b',6)]:
    h=abs(x[column])*5e-4
    energies=[];temperatures=[]
    for sign in [-1,1]:
        p=dict(ad.GM1);p[key]+=sign*h
        root=brentq(lambda t:ad.state(n,t,Y,params=p).entropy_density-entropy,
                    .98*T,1.02*T,xtol=2e-10,rtol=2e-12)
        s=ad.state(n,root,Y,params=p)
        energies.append(s.energy_density);temperatures.append(root)
    fd=(energies[1]-energies[0])/(2*h)
    expected=diag['hellmann_feynman_free_energy_derivatives'][column-3]
    add('independent entropy-root de/dtheta|s,n '+key,
        abs(fd-expected)/max(1.,abs(expected)),2e-6)
    tfd=(temperatures[1]-temperatures[0])/(2*h)
    texpected=-J[11,column]/J[11,1]
    add('independent entropy-root dT/dtheta|s,n '+key,
        abs(tfd-texpected)/max(1.,abs(texpected)),2e-6)

# Raw C derivatives differ from g derivatives: C=g²/m_mes², fixed meson mass.
# Check exact vector-work chain using g=sqrt(C) with an arbitrary unit meson scale.
g=math.sqrt(x[4]);hg=1e-4*g
states=[]
for sign in [-1,1]:
    p=dict(ad.GM1);p['C_omega']=(g+sign*hg)**2
    states.append(ad.state(n,T,Y,params=p).pressure)
fd=(states[1]-states[0])/(2*hg)
expected=2*g*J[9,4]
add('raw C-to-g vector-pressure chain at fixed meson scale',abs(fd/expected-1),1e-9)

out=dict(scope='Independent entropy-root paths and raw-parameter chain; not external EOS validation',
         checks=checks,all_passed=all(c['passed'] for c in checks))
Path(__file__).with_name('parameter_ad_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
