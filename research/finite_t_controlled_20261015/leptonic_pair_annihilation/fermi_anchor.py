"""Primary2006Eqs20-25 separable massive FD anchor, independent thermal coordinates."""
from pathlib import Path
import json,math
from scipy.integrate import quad
from scipy.special import expit
import pair as P
H=Path(__file__).parent;checks=[];rows=[];T=2.;mu=.17446333200826397;m=P.ME
cache={}
def G(r,sign):
 if (r,sign)not in cache:cache[(r,sign)]=quad(lambda p:p*p*math.hypot(m,p)**(2*r)*expit((sign*mu-math.hypot(m,p))/T),0,200,epsabs=1e-13,epsrel=3e-12)[0]/m**(3+2*r)
 return cache[(r,sign)]
def M(n,j,C,sign):
 v,a=C.cv,C.ca;n=n/2;j=j/2
 return (7*v*v-2*a*a)*G(n-.5,sign)*G(j-.5,-sign)+9*v*v*G(n,sign)*G(j,-sign)+(v*v+a*a)*(4*G(n+.5,sign)*G(j+.5,-sign)-G(n-.5,sign)*G(j+.5,-sign)-G(n+.5,sign)*G(j-.5,-sign))
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
data=json.loads((H/'thermal_results.json').read_text())
for flavor in ['e','mu']:
 C=P.Couplings(flavor);rate=P.GF**2*m**8/(18*math.pi**5)*M(0,0,C,1)/(P.HBAR*P.HC**3)
 energy=P.GF**2*m**9/(18*math.pi**5)*(M(1,0,C,1)+M(1,0,C,-1))/(P.HBAR*P.HC**3)
 reference=next(z['result']for z in data['rows']if z.get('label')==flavor and z['result']['controls']['radial']==20)
 gate('primary massive separated FD event '+flavor,reference['totals']['number']['value']/rate-1,3e-10)
 q=reference['totals']['nu_energy']['value']+reference['totals']['antinu_energy']['value']
 gate('primary massive separated FD total energy '+flavor,q/energy-1,3e-10)
 rows.append(dict(flavor=flavor,primary_M00=M(0,0,C,1),event_rate=rate,total_energy=energy))
(H/'fermi_anchor_results.json').write_text(json.dumps(dict(checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
