from pathlib import Path
import json,math
import numpy as np
from numpy.polynomial.legendre import leggauss
import moments as M
P=Path(__file__).resolve().parent;checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
b,meta=M.PC.matched(2,1e-7,.3)
for name,kwargs in [('emptyScenario',dict(scenarios=[])),('booln',dict(neutron_order=True)),('loworder',dict(outgoing_order=3)),('nantail',dict(tail=math.nan)),('hightail',dict(tail=101)),('nanbudget',dict(budget_seconds=math.nan)),('overbudget',dict(budget_seconds=121)),('radNaN',dict(scenarios=[dict(name='a',mode='FD',mu_nu=math.nan)])),('radextreme',dict(scenarios=[dict(name='a',mode='FD',mu_nu=30)])),('unknownradiation',dict(scenarios=[dict(name='a',mode='MB')]))]:
 try:M.integrate(b,**kwargs);gate('reject '+name,1,0)
 except ValueError:gate('reject '+name,0,0)
for bad in [M.PC.crossed.Bath(2,950,950,0),M.PC.crossed.Bath(.1,900,900,0),M.PC.crossed.Bath(2,900,900,10)]:
 try:M.integrate(bad);gate('reject selected domain',1,0)
 except ValueError:gate('reject selected domain',0,0)
# Vacuum reverse-2body reciprocity, absolute source-polynomial cross section.
for Ee in [1.,3.,12.]:
 s=M.MN*M.MN+M.ME*M.ME+2*M.MN*Ee;rt=math.sqrt(s);pi=math.sqrt((s-(M.MN+M.ME)**2)*(s-(M.MN-M.ME)**2))/(2*rt);pf=(s-M.MP*M.MP)/(2*rt);En=(s+M.MN*M.MN-M.ME*M.ME)/(2*rt);El=rt-En
 x,w=leggauss(24);N=len(x);pn=np.tile([En,0,0,pi],(N,1));l=np.tile([El,0,0,-pi],(N,1));k=np.column_stack([np.full(N,pf),pf*np.sqrt(1-x*x),np.zeros(N),pf*x]);p=np.column_stack([np.full(N,(s+M.MP*M.MP)/(2*rt)),-k[:,1:]])
 eb=M.E.Bath(2,M.MP,M.MN,900,900,0,ml=M.ME,anti=True,Mweak=M.MW);amp=M.E.amplitude(eb,k,p,l,pn)/2;sigma=float(w@amp/(32*math.pi*s)*pf/pi)*M.HC**2*1e-26
 antiE=(s-M.MP*M.MP)/(2*M.MP);ch=M.PC.V.Channel(M.MP,M.MN,M.ME,True);inverse=M.PC.K.C.total(ch,antiE,order=24,method='source');expected=inverse*.5*(pf/pi)**2
 gate('absolute vacuum detailed cross-section balance',sigma/expected-1,3e-10);rows.append(dict(Epositron_resttarget=Ee,sigma_forward_cm2=sigma,source_inverse_sigma_cm2=inverse,source_expected_forward_cm2=expected,spin_ratio=.5))
(P/'controls_anchor.json').write_text(json.dumps(dict(checks=checks,rows=rows,passed=all(c['passed']for c in checks)),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks));print([c for c in checks if not c['passed']])
