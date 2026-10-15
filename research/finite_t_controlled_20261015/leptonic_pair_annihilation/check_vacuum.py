"""Fresh massive traces versus primary scalar amplitude and derived cross section."""
from pathlib import Path
import math,json
import numpy as np
from numpy.polynomial.legendre import leggauss
import pair as P
H=Path(__file__).parent;checks=[];rows=[];source_diagnostics=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
for flavor in ['e','mu','tau']:
 C=P.Couplings(flavor)
 for E in [.51101,.6,2.,10.]:
  s=4*E*E;xx,ww=leggauss(12);total=0.
  for c,w in zip(xx,ww):
   v=P.cm(s,float(c));trace=float(P.amplitude(*v,C,method='trace')[0]);scalar=float(P.amplitude(*v,C)[0]);gate('trace/scalar '+flavor+' '+str(E)+' '+str(float(c)),trace/scalar-1,3e-10)
   total+=w*trace/(32*math.pi*s)/math.sqrt(1-4*P.ME**2/s)
  gate('derived exact CM total '+flavor+' '+str(E),total/P.sigma(s,C)-1,3e-11)
  rows.append(dict(flavor=flavor,electron_CM_energy=E,sigma_natural=total,sigma_cm2=total*P.HC**2*1e-26))
 # SourceEq17 as printed in HTML AND PDF text differs in its finite-mass D
 # term from angular integration of the same paper's Eq11. Preserve diagnosis.
 E=.6;s=4*E*E;pm,pp,_,_=P.cm(s,0.);d=float(P.dot(pm,pp)[0]);m=P.ME;v,a=C.cv,C.ca
 printed=P.GF**2/(12*math.pi*E*E)*((v*v+a*a)*(m**4+3*m*m*d+2*d*d)+(v*v-a*a)*(m**4+2*m*m*d))
 derived=float(P.sigma_v(pm,pp,C)[0])
 source_diagnostics.append(dict(flavor=flavor,case='massive Ecm .6',printed_2006Eq17_sigma_v=printed,integrated_Eq11_trace_sigma_v=derived,relative_discrepancy=printed/derived-1,qualified_as_cross_section_anchor=False))
 # Independent massless high-energy total and NR vector-only threshold limit.
 s=400.;gate('massless total '+flavor,P.sigma(s,C,mass=0)/(P.GF**2*s*(C.cv**2+C.ca**2)/(12*math.pi))-1,2e-14)
 s=4*P.ME**2*(1+1e-8);beta=math.sqrt(1-4*P.ME**2/s)
 gate('threshold vector sigma*v '+flavor,P.sigma(s,C)*2*beta/(P.GF**2*P.ME**2*C.cv**2/math.pi)-1,3e-6)
gate('e flavor CC shift vector',P.Couplings('e').cv-P.Couplings('mu').cv-1,2e-16)
gate('e flavor CC shift axial',P.Couplings('e').ca-P.Couplings('mu').ca-1,0)
(H/'vacuum_results.json').write_text(json.dumps(dict(checks=checks,rows=rows,all_pass=all(c['passed']for c in checks),preserved_source_conflict=source_diagnostics),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']],source_diagnostics=source_diagnostics)))
