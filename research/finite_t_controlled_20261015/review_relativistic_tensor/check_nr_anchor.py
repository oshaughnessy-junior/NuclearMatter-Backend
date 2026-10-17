"""New matched point-V/A limit, not a full-current/contact comparison."""
from pathlib import Path
import sys,json,math,numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'relativistic_delta_oracle'));import oracle as o
sys.path.insert(0,str(ROOT/'relativistic_tensor'));import tensor as t
sys.path.insert(0,str(ROOT/'response_reference'));from free_density_response import PairState,response,HBARC
rows=[];checks=[]
for M in (1e6,2e6):
 T=3.;E=16.;c=.37;q0=math.sqrt(2*E*E*(1-c));w=.2*q0*math.sqrt(T/M)+q0*q0/(2*M);El=E-w;ml=.51099895069;mul=2.4
 p=o.Point(T,M,M,M-8.,M-9.,E,El,c,ml,mul,False,M);r=o.integrate(p,40,16,80,current_model='point_VA');sp=t.State(T,M,M,M-8.,M-9.,M);model=t.C.FormFactors(weak_magnetism=False,pseudoscalar=False,q2_dependence=False);a=t.differential(sp,E,El,c,ml,mul,model=model,energy_order=40,phi_order=16,tail=80.)
 pl=math.sqrt(El*El-ml*ml);S=response(PairState(T,M,M,M,M-8.,M-9.),r['q'],w);angular=(1+3*1.2723**2)+(1-1.2723**2)*(pl/El)*c;chi=o.GF**2*o.VUD**2*HBARC**3/(4*math.pi**2*o.HBAR)*El*pl*angular*S*o.expit((El-mul)/T)
 error=r['chi']/chi-1;agree=a['chi']['value']/r['chi']-1
 checks.append(dict(name='fresh matchedNR absolute M'+str(M),error=error,tolerance=3e-5,passed=abs(error)<3e-5));checks.append(dict(name='fresh matchedNR author/oracle M'+str(M),error=agree,tolerance=3e-8,passed=abs(agree)<3e-8));rows.append(dict(mass=M,Enu=E,El=El,w=w,q=r['q'],NR_allowed=chi,oracle=r['chi'],author=a['chi']['value'],relative_limit_difference=error))
checks.append(dict(name='fresh controlledNR approach',passed=abs(rows[1]['relative_limit_difference'])<.6*abs(rows[0]['relative_limit_difference'])))
(P/'nr_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2,default=lambda x:x.item())+'\n');print(len(checks),all(c['passed']for c in checks),rows)
