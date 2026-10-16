"""PC input/log/operator controls and isolated spectral controls."""
from pathlib import Path
import importlib.util,sys,json,math,time
import numpy as np
from scipy.special import expit
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('hotPC_control_function',P/'spectrum.py');S=importlib.util.module_from_spec(s);sys.modules[s.name]=S;s.loader.exec_module(S)
def run(isolate=False):
 start=time.monotonic();checks=[];rows=[];b,_=S.matched()
 def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
 for E in [float('nan'),float('inf'),True,0.,-1.,60.1]:
  try:S.integrate(E);ok=False
  except ValueError:ok=True
  gate('incidentguard '+str(E),0 if ok else 1,0)
 for k,v in [('lepton_order',True),('angle_order',4.1),('energy_order',7),('phi_order',True),('tail',float('nan')),('outer_tail',float('inf')),('budget_seconds',float('nan'))]:
  try:S.integrate(5.,**{k:v});ok=False
  except ValueError:ok=True
  gate('numericalguard '+k,0 if ok else 1,0)
 wrong=S.C.Bath(b.T,b.mu_n,b.mu_p,b.mu_e+.1)
 try:S.differential(wrong,5.,3.7,.29);ok=False
 except ValueError:ok=True
 gate('changedmatterrejected',0 if ok else 1,0)
 for log,expected in [(None,'empty_support'),(-1000.,'underflow'),(1000.,'overflow')]:
  r=S.represented(log);gate('ordinarylogrepresentation '+str(log),0 if r['reason']==expected and r['log']==log else 1,0)
 if isolate:
  data=json.loads((P/'results.json').read_text());base=data['rows'][0]['rules'][-1]
  for name,change in [('inner_tail',dict(tail=90.)),('outer_tail',dict(outer_tail=50.)),('azimuth',dict(phi_order=12))]:
   controls=dict(lepton_order=12,angle_order=20,energy_order=16,phi_order=8,tail=70.,outer_tail=40.);controls.update(change);r=S.integrate(2.,**controls);rows.append(dict(kind=name,result=r))
   for k in ['j','chi']:gate(name+' '+k,r[k]['value']/base[k]['value']-1,3e-6)
  for row in data['rows']:
   r=row['rules'][-1];E=row['E'];j=r['j']['value'];chi=r['chi']['value'];eq=r['equilibrium_mu_antinue']
   for shift in [-.8,.6]:
    mu=eq+shift;f=float(expit((mu-E)/2));F=j*(1-f);B=chi*f;R=F-B;Q=E*R;Le_m=R;Le_rad=-R;sm=(-Q+eq*R)/2;sr=(Q-mu*R)/2;entropy=(F-B)*math.log(F/B)
    gate('entropypositive '+str((E,shift)),0 if entropy>=0 else 1,0);gate('entropydecomposition '+str((E,shift)),(sm+sr-entropy)/(F+B),3e-12);gate('Le '+str((E,shift)),(Le_m+Le_rad)/(F+B),0)
 out=dict(checks=checks,rows=rows,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],seconds=time.monotonic()-start);(P/('isolated_controls.json'if isolate else'controls_results.json')).write_text(json.dumps(out,indent=2)+'\n');print({k:out[k]for k in ['passed','failed','seconds']})
if __name__=='__main__':run('--isolate'in sys.argv)
