"""Supplement the repaired review with independent adaptive event entropy."""
from pathlib import Path
import sys,json,math
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'neutral_current'))
import free_nc as n
s=n.state_density(1.3,.0013,offset=17.)
pair=n.pair_from_species(s);couplings=n.Couplings(.2,.7)
E=np.array([3.5,5.,7.5,11.,18.]);w=np.array([.3,1.3,.8,2.1,4.]);f=np.array([.91,.07,.63,.21,.005]);g=n.DOS*E**2*w
r=n.discrete_collision(s,E,w,f,couplings,angle_order=40)
entropy=0.;bath=0.
for i in range(5):
    for j in range(i):
        forward=g[i]*w[j]*quad(lambda c:n.kernel(pair,E[i],E[j],c,couplings),-1,1,epsabs=1e-60,epsrel=2e-12)[0]*f[i]*(1-f[j])
        back=g[j]*w[i]*quad(lambda c:n.kernel(pair,E[j],E[i],c,couplings),-1,1,epsabs=1e-60,epsrel=2e-12)[0]*f[j]*(1-f[i])
        net=forward-back
        entropy+=net*math.log(forward/back)
        bath+=(E[i]-E[j])*net
macro=-r['number_source']@(np.log(f)-np.log1p(-f)+E/s.T)
path=Path(__file__).with_name('neutral_current_repair_verification.json')
report=json.loads(path.read_text())
new=[]
for name,a,b in [('independent adaptive paired entropy regression',macro,entropy),('independent adaptive paired bath energy regression',r['net_neutrino_energy_source'],-bath)]:
    error=abs(a/b-1)
    new.append(dict(name=name,actual=float(a),expected=float(b),error=float(error),passed=bool(error<2e-11)))
old=json.loads(path.with_name('neutral_current_refined_initial.json').read_text())
new.append(dict(name='valid moments unchanged by control/pair repair',passed=report['samples']==old['samples']))
names={c['name'] for c in new}
report['checks']=[c for c in report['checks'] if c['name'] not in names]+new
report['valid_selected_cases_passed']=all(c['passed'] for c in report['checks'])
path.write_text(json.dumps(report,indent=2)+'\n')
print(len(report['checks']),'checks;',report['valid_selected_cases_passed'])
assert report['valid_selected_cases_passed']
