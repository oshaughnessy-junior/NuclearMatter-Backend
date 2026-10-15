from pathlib import Path
import json,math
from scipy.integrate import quad
import scatter as S
P=Path(__file__).resolve().parent;checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for flavor in ['e','mu']:
 c=S.P.Couplings(flavor);errors=[]
 for ratio in [1e-3,5e-4,2.5e-4]:
  E=ratio*S.ME;tm=2*E*E/(S.ME+2*E);sig=quad(lambda z:S.vacuum_dsigma_drecoil(E,z,c),0,tm,epsabs=1e-60,epsrel=1e-12)[0];lim=S.GF**2*E*E/math.pi*(c.cv*c.cv+3*c.ca*c.ca);errors.append(abs(sig/lim-1));rows.append(dict(flavor=flavor,E_over_me=ratio,sigma=sig,NRleading=lim,error=errors[-1]))
 gate('NRlimit final '+flavor,errors[-1],.002);gate('NRlimit firstorder convergence '+flavor,errors[-1]/errors[-2]-.5,.01)
 E=1000*S.ME;tm=2*E*E/(S.ME+2*E);sig=quad(lambda z:S.vacuum_dsigma_drecoil(E,z,c),0,tm,epsabs=1e-50,epsrel=1e-12)[0];high=S.GF**2*S.ME*E/(2*math.pi)*((c.cv+c.ca)**2+(c.cv-c.ca)**2/3);gate('relativistic targetrest limit '+flavor,sig/high-1,.002)
sw=.23129;c=S.P.Couplings('mu');gate('primaryGuo40 λ1',(c.cv+c.ca)**2/(1-4*sw+4*sw**2)-1,3e-15);gate('primaryGuo40 λ2',(c.cv-c.ca)**2/(4*sw**2)-1,3e-15);gate('primaryGuo40 λ3',-(c.cv*c.cv-c.ca*c.ca)/(2*sw-4*sw**2)-1,3e-15)
(P/'limit_results.json').write_text(json.dumps(dict(checks=checks,rows=rows,passed=all(c['passed']for c in checks),meaning='formal Fermi-model NR and highenergy vacuumlimits, not extended thermal-domain certificate'),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks));print([c for c in checks if not c['passed']])
