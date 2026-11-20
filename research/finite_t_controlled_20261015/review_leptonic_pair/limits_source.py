from pathlib import Path
import sys,json,math
from scipy.special import zeta
from numpy.polynomial.legendre import leggauss
import reference as F
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'leptonic_pair_annihilation'));import pair as A
checks=[];rows=[];diagnostics=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for flavor in ['e','mu']:
 c=A.Couplings(flavor);vals=[]
 for T in [1.,2.]:
  r=F.separated(T,0.,c.cv,c.ca,m=0.);v=c.cv*c.cv+c.ca*c.ca;expectedR=49*math.pi**3/64800*F.GF**2*v*T**8/(F.HBAR*F.HC**3);expectedQ=7*float(zeta(5))/(12*math.pi)*F.GF**2*v*T**9/(F.HBAR*F.HC**3)
  gate('massless FD absolute event '+flavor,r['number']/expectedR-1,3e-11);gate('massless FD absolute total energy '+flavor,(r['nu_energy']+r['antinu_energy'])/expectedQ-1,3e-11);gate('massless equal partners',r['nu_energy']/r['antinu_energy']-1,0);vals.append(r);rows.append(dict(flavor=flavor,T=T,result=r))
 gate('massless number T8',vals[1]['number']/vals[0]['number']/256-1,3e-12);gate('massless energy T9',vals[1]['nu_energy']/vals[0]['nu_energy']/512-1,3e-12)
 p=F.separated(1.73,.29,c.cv,c.ca);m=F.separated(1.73,-.29,c.cv,c.ca)
 for key,partner in [('number','number'),('nu_energy','antinu_energy'),('electron_energy','positron_energy')]:gate('CP species exchange '+key,p[key]/m[partner]-1,2e-13)
 # Literal printed2006Eq17 is an EXCLUDED diagnostic, not silently corrected.
 for ee in [.6,.89]:
  ss=4*ee*ee;pm,pp,k,a=F.cm(ss,.17);d=F.dot(pm,pp);mass=F.ME;v,g=c.cv,c.ca;printed=F.GF**2/(12*math.pi*ee**2)*((v*v+g*g)*(mass**4+3*mass*mass*d+2*d*d)+(v*v-g*g)*(mass**4+2*mass*mass*d));xx,ww=leggauss(24);beta=math.sqrt(1-mass*mass/ee**2);integrated=sum(w*F.trace(*F.cm(ss,float(x)),v,g)/(16*math.pi*ss)for x,w in zip(xx,ww));difference=F.GF**2/(12*math.pi*ee**2)*(v*v-g*g)*mass*mass*(2*mass*mass+d)
  gate('diagnosed Eq11-vs-Eq17 polynomial difference',(integrated-printed)/difference-1,3e-10);diagnostics.append(dict(flavor=flavor,Ecm=ee,printed_sigma_v=printed,trace_angular_sigma_v=integrated,relative_conflict=printed/integrated-1,accepted_anchor=False))
for name,fn in [('nanT',lambda:A.thermal(math.nan,0,A.Couplings('e'))),('boolorder',lambda:A.thermal(2,0,A.Couplings('e'),incident=True)),('mixedRadT',lambda:A.thermal(2,0,A.Couplings('e'),radiation=A.Radiation(2.1,0,0))),('onepartnerNaN',lambda:A.Radiation(2,0,math.nan)),('badflavor',lambda:A.Couplings('x')),('badmixingangle',lambda:A.Couplings('e',math.inf)),('nantail',lambda:A.thermal(2,0,A.Couplings('e'),tail=math.nan))]:
 try:fn();gate('reject '+name,1,0)
 except ValueError:gate('reject '+name,0,0)
(P/'limits_source_results.json').write_text(json.dumps(dict(checks=checks,passed=all(c['passed']for c in checks),massless_rows=rows,EXCLUDED_printed_source_formula=diagnostics,source_url='https://arxiv.org/html/astro-ph/0511555',source_interpretation='Eq6-11 trace independently supports CM amplitude. PrintedEq17 conflicts at finite mass; no erratum asserted. Source number emissivity F=2R, event R must not be doubled again.'),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks));print([c for c in checks if not c['passed']])
