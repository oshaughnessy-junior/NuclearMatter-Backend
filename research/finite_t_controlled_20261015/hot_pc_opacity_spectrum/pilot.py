"""Absolute crossed-current/phase anchors gate every later PC spectrum claim."""
from pathlib import Path
import importlib.util,sys,json,math,time
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True

def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
S=load('hotPC_spectrum_pilot_function',P/'spectrum.py');O=load('hotPC_original_delta',R/'relativistic_delta_oracle/oracle.py')
# Load only purecomplexcurrent functions and restore generic importstate.
oldpath=list(sys.path);aliases=['tensor','kernel','free_baryons'];sentinel=object();old={n:sys.modules.get(n,sentinel)for n in aliases}
try:
 sys.modules['tensor']=S.K;sys.modules['free_baryons']=S.C.EOS
 T=load('hotPC_crossed_pure_kernel',R/'relativistic_crossed_thermal/kernel.py');sys.modules['kernel']=T
 PC=load('hotPC_literal_complex_function',R/'positron_capture/pc_kernel.py')
finally:
 sys.path[:]=oldpath
 for k,v in old.items():
  if v is sentinel:sys.modules.pop(k,None)
  else:sys.modules[k]=v

def run():
 start=time.monotonic();checks=[];points=[];b,meta=S.matched();tb=T.Bath(**vars(b))
 def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
 for en,el,co in [(2.,.71,-.4),(5.,3.7,.29),(10.,8.69,-.36)]:
  actual=S.differential(b,en,el,co,energy_order=24,phi_order=8,tail=80.);literal=PC.differential(tb,en,el,co,energy_order=28,phi_order=8,tail=85.)
  point=O.Point(b.T,S.C.MP,S.C.MN,b.mu_p,b.mu_n,en,el,co,S.C.ME,-b.mu_e,True)
  refs=[O.integrate(point,order=24,nphi=6,tail=65.),O.integrate(point,order=36,nphi=10,tail=85.)]
  for k in ['j','chi']:
   gate('antidelta refinement '+str((en,k)),refs[0][k]/refs[1][k]-1,3e-9);gate('absolute antidelta '+str((en,k)),actual[k]['value']/refs[1][k]-1,3e-10);gate('literal vtrace tensor '+str((en,k)),actual[k]['value']/literal[k]['value']-1,3e-10)
  gate('anti KMS AFTER positivebranches '+str(en),actual['j']['log']-actual['chi']['log']+(en-(b.mu_n-b.mu_p-b.mu_e))/b.T,3e-12)
  points.append(dict(input=[en,el,co],actual=actual,literal=literal,original_delta=[{k:r[k]for k in ['j','chi','order','nphi','tail']}for r in refs]))
 # Timelike q is past-directed for masslessincoming/massiveoutgoing lepton.
 # Such transfer cannot convert lighter positiveenergy p into heavier n.
 for en,el,co in [(2.,3.29,.999),(5.,6.293,.9999)]:
  actual=S.differential(b,en,el,co,energy_order=24,phi_order=8,tail=80.);p=math.sqrt(el*el-S.C.ME**2);q2=(en-p)**2+2*en*p*(1-co);w=en-el
  gate('timelikepast kinematic witness '+str(en),0 if w<0 and w*w>q2 else 1,0)
  gate('exactempty timelikep-to-n '+str(en),0 if actual['support']==[]and actual['j']['log']is None and actual['chi']['log']is None and actual['j']['value']==actual['chi']['value']==0 else 1,0)
  points.append(dict(input=[en,el,co],actual=actual,timelike_past=True))
 out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],points=points,matched_state=meta,seconds=time.monotonic()-start)
 (P/'point_results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k]for k in ['passed','failed','seconds']})
 if out['failed']:raise ArithmeticError('Absolute crossed anchor failure; no spectral claim')
if __name__=='__main__':run()
