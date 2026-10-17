"""New direct-cosine/sinh-transfer/target-energy moment reference near signed zero."""
from pathlib import Path
import math,time,json
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
HERE=Path(__file__).resolve().parent;T=2.;MU=.17446333200826397;M=.51099895069;GF=1.1663785e-11;HB=6.582119569e-22
def D(E,F,c,tail):
 om=E-F;z=2*E*F*(1-c);Q=math.sqrt(om*om+z);lo=(-om+Q*math.sqrt(1+4*M*M/z))/2;hi=max(lo,abs(om),MU)+tail*T
 kz=(E*E+Q*Q-F*F)/(2*Q);kx2=max(0.,E*E-kz*kz);h=z/2;v=.96258;a=.5;L,R=(v+a)**2,(v-a)**2
 def target(e):
  pz=(2*e*om-z)/(2*Q);pt2=max(0.,e*e-M*M-pz*pz);u=E*e-kz*pz;vv=kx2*pt2/2
  mm=M*M*(v*v-a*a)*h
  minus=(L*(u*u+vv)+R*((u-h)**2+vv)-mm)*expit((MU-e)/T)*expit((e+om-MU)/T)
  plus=(R*(u*u+vv)+L*((u-h)**2+vv)-mm)*expit((-MU-e)/T)*expit((e+om+MU)/T)
  return minus+plus
 value=quad(target,lo,hi,epsabs=2e-9,epsrel=3e-10,limit=120)[0]
 return F*GF*GF/(4*math.pi**3*E*Q*HB)*value
def integrate(E,order,angle,upper,tail,lower=0.):
 started=time.monotonic();c,w=leggauss(angle);x,g=leggauss(order);out=np.zeros(4)
 for ca,wa in zip(c,w):
  width=E*math.sqrt(2*(1-ca));low=math.asinh((lower-E)/width);high=math.asinh((upper-E)/width)
  cuts=sorted(set([low,high]+[a for a in [-3.,-2.,-1.,-.3,0.,.3,1.,2.,3.]if low<a<high]))
  for a,b in zip(cuts[:-1],cuts[1:]):
   for t,wt in zip((a+b)/2+(b-a)*x/2,g*(b-a)/2):
    if time.monotonic()-started>110:raise TimeoutError('Independent moment integration bounded110s')
    f=E+width*math.sinh(t);jac=width*math.cosh(t);d=D(E,f,ca,tail)*jac*wt*wa
    out+=d*np.array([1.,1-ca,f,f-E])
 return dict(E=E,moments=dict(zip(['rate','transport_rate','outgoing_energy_moment','neutrino_energy_change'],out.tolist())),controls=dict(order=order,angle=angle,upper=upper,lower=lower,tail=tail),seconds=time.monotonic()-started)
checks=[];rows=[];E=8.19
for controls in [(18,24,100.,65.),(26,36,140.,90.)]:
 row=integrate(E,*controls);rows.append(row)
 (HERE/'phase_partial.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
ref=next(r['fine_actual_fallback']for r in json.loads((HERE/'results.json').read_text())['fresh']if r['E']==E)
for key in ['rate','transport_rate','outgoing_energy_moment','neutrino_energy_change']:
 scale=ref['rate']*T if key=='neutrino_energy_change'else ref[key]
 for name,a,b in [('own moment refinement',rows[0]['moments'][key],rows[1]['moments'][key]),('independent phase vs actualCM',rows[1]['moments'][key],ref[key])]:
  error=(a-b)/scale;checks.append(dict(name=name+' '+key,error=error,tolerance=1e-4,passed=abs(error)<=1e-4))
tail=integrate(E,16,24,140.,80.,lower=100.)
for key in tail['moments']:
 scale=ref['rate']*T if key=='neutrino_energy_change'else ref[key];error=tail['moments'][key]/scale
 checks.append(dict(name='actual outgoing100..140 tail '+key,error=error,tolerance=1e-8,passed=abs(error)<=1e-8))
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,tail=tail,method='analytic target-azimuth invariant polynomial, adaptive targetENERGY, Gauss directcosine and sinh(F-E) outgoingenergy; signedintegrand accumulated directly, no fit orauthorCM/D/phasefunctions called',scope='one new nearzero energy at fixedbath; no uniform certificate or physicaluncertainty')
(HERE/'phase_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ['passed','failed']}))
