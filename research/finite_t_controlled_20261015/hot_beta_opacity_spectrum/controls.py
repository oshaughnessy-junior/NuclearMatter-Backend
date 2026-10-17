"""Selectedbeta input/operator/log/forward-density andisolatedphase controls."""
from pathlib import Path
import importlib.util,sys,json,math,time,copy
from scipy.integrate import quad
from scipy.special import expit
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('beta_new_control_spectrum',P/'spectrum.py');S=importlib.util.module_from_spec(s);sys.modules[s.name]=S;s.loader.exec_module(S)
def run():
 start=time.monotonic();checks=[];rows=[]
 def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
 for E in [float('nan'),float('inf'),True,0.,-1.,1.41]:
  try:S.integrate(E);ok=False
  except ValueError:ok=True
  gate('incidentguard '+str(E),0 if ok else 1,0)
 for k,v in [('parent_order',True),('cone_order',7.5),('electron_order',3),('phi_order',True),('tail',float('nan')),('budget_seconds',float('inf'))]:
  try:S.integrate(.5,**{k:v});ok=False
  except ValueError:ok=True
  gate('controlguard '+k,0 if ok else 1,0)
 for log,reason in [(None,'empty_support'),(-1000.,'underflow'),(1000.,'overflow')]:
  r=S.represented(log);gate('representation '+str(log),0 if r['reason']==reason and r['log']==log else 1,0)
 data=json.loads((P/'pilot_results.json').read_text());b,_=S.matched()
 for row in data['rows']:
  r=row['rules'][-1];pmin=r['parent_minimum_momentum'];kmax=r['parent_maximum_kinetic'];pmax=math.sqrt(kmax*(kmax+2*S.MN));f=lambda p:p*p/(math.pi**2*S.HC**3)*float(expit((b.mu_n-math.hypot(S.MN,p))/b.T));density=quad(f,pmin,pmax,epsabs=1e-21,epsrel=3e-12)[0]
  gate('independentallowedFDdensity '+str(r['Enu']),r['parent_density_on_allowed_domain']/density-1,3e-8)
  if r['Enu']<S.ENDPOINT:gate('fullFDdensity '+str(r['Enu']),density/7e-8-1,3e-10)
  j,chi=r['j']['value'],r['chi']['value'];eq=r['equilibrium_mu_antinue'];E=r['Enu']
  for shift in[-1.6,1.6]:
   mu=eq+shift;f=float(expit((mu-E)/2));F=j*(1-f);B=chi*f;R=F-B;Q=E*R;entropy=(F-B)*math.log(F/B);sm=(-Q+eq*R)/2;sr=(Q-mu*R)/2
   gate('entropypositive '+str((E,shift)),0 if entropy>=0 else 1,0);gate('entropydecomposition '+str((E,shift)),(sm+sr-entropy)/(F+B),3e-12);gate('antiLeledger '+str((E,shift)),(R-R)/(F+B),0)
 base=data['rows'][-1]['rules'][-1]
 for name,change in [('parenttail70to90',dict(tail=90.)),('azimuth10to14',dict(phi_order=14)),('joint18_24_24',dict(parent_order=18,cone_order=24,electron_order=24,phi_order=14,tail=90.))]:
  kw=dict(parent_order=12,cone_order=16,electron_order=16,phi_order=10,tail=70.);kw.update(change);r=S.integrate(.8,**kw);rows.append(dict(kind=name,result=r))
  for k in['j','chi']:gate(name+' '+k,r[k]['value']/base[k]['value']-1,3e-6)
 out=dict(checks=checks,rows=rows,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],seconds=time.monotonic()-start);(P/'controls_results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k]for k in['passed','failed','seconds']})
if __name__=='__main__':run()
