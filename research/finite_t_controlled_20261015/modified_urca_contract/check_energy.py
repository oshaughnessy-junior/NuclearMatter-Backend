from pathlib import Path
import json,math,time
from scipy.integrate import quad
import energy_reference as E
P=Path(__file__).resolve().parent;checks=[];rows=[];start=time.monotonic()
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for x in [-12.,-3.,0.,.7,4.,15.]:
 direct2=quad(lambda t:E.f(t)*E.f(x-t),-90,90,epsabs=2e-12,epsrel=2e-11,points=sorted(set([0.,x])))[0];gate('2-Fermi original convolution',direct2/E.J2(x)-1,2e-10)
 c3=E.convolution3(x);gate('3-Fermi convolution',c3/E.J3(x)-1,2e-10)
 vals=[E.convolution5(x,L)for L in [60.,90.]];gate('5-Fermi independent convolution',vals[-1]/E.J5(x)-1,2e-10);gate('5-Fermi tail refinement',vals[0]/vals[1]-1,2e-10)
 gate('five-hole detailed balance',E.J5(-x)*math.exp(-x)/E.J5(x)-1,2e-12)
 rows.append(dict(x=x,direct2=direct2,direct3=c3,convolution5=vals,polynomial5=E.J5(x)))
analytic=E.equilibrium_analytic();numerical=E.moments()
for key in ['number','energy']:gate('equilibrium dimensionless '+key,numerical[key]/analytic[key]-1,2e-11)
# An independently ordered outer y integral of the convolution, no polynomialJ5.
conv={key:quad(lambda y:y**power*E.convolution5(y),0,90,epsabs=1e-7,epsrel=2e-8,limit=120)[0]for key,power in [('number',2),('energy',3)]}
for key in conv:gate('outer-integrated independent '+key,conv[key]/analytic[key]-1,3e-8)
# Dimensional T power check with a FROZEN unspecified angular/current prefactor.
for Ta,Tb in [(.01,.02),(.12,.3)]:
 gate('numberT7',(Tb**7*analytic['number'])/(Ta**7*analytic['number'])/(Tb/Ta)**7-1,2e-15);gate('energyT8',(Tb**8*analytic['energy'])/(Ta**8*analytic['energy'])/(Tb/Ta)**8-1,2e-15)
# Neutrino/antineutrino emitted branch enhancement; no inverse absorption claim.
non=[]
for a in [.1,1.,4.]:
 plus,minus=E.moments(a),E.moments(-a);Q=(plus['energy']+minus['energy'])/(2*analytic['energy']);expected=1+22020*a*a/(11513*math.pi**2)+5670*a**4/(11513*math.pi**4)+420*a**6/(11513*math.pi**6)+9*a**8/(11513*math.pi**8)
 gate('two-emitted-branch energy polynomial',Q/expected-1,3e-10);gate('chemical source direction',int(plus['number']<=minus['number']),0);non.append(dict(affinity=a,positive=plus,negative=minus,energy_enhancement=Q))
for fn in [lambda:E.J5(math.nan),lambda:E.J2(math.inf),lambda:E.moments(11.)]:
 try:fn();gate('finite/domain reject',1,0)
 except ValueError:gate('finite/domain reject',0,0)
out=dict(checks=checks,passed=all(c['passed']for c in checks),rows=rows,equilibrium_analytic=analytic,numerical=numerical,independent_convolution_integrals=conv,nonequilibrium=non,mean_Enu_over_T=analytic['energy']/analytic['number'],seconds=time.monotonic()-start,scope='dimensionless Fermi-surface ENERGY-ONLY benchmark; physical angular/current/DOS prefactor absent')
(P/'energy_results.json').write_text(json.dumps(out,indent=2)+'\n');print(len(checks),out['passed'],out['seconds'],out['mean_Enu_over_T']);print([c for c in checks if not c['passed']])
