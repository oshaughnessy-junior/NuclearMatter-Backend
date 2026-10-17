"""Absolute differential and first bounded spectral pilot, before broad work."""
from pathlib import Path
import importlib.util,sys,json,math,hashlib
P=Path(__file__).parent;ROOT=P.parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
S=load('hot_EC_spectrum_pilot_function',P/'spectrum.py');O=load('hot_EC_independent_delta_oracle',ROOT/'relativistic_delta_oracle/oracle.py');b,meta=S.matched();checks=[];points=[]
def ck(n,e,g=3e-10):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
for nu,el,c in[(2.,3.29,-.41),(5.,6.293,.37),(10.,11.28,-.23),(2.,3.29,.999)]:
 d=S.differential(b,nu,el,c,energy_order=20,phi_order=8,tail=70.);state=S.K.State(b.T,S.C.MN,S.C.MP,b.mu_n,b.mu_p,S.C.MW);ref=S.K.differential(state,nu,el,c,S.C.ME,b.mu_e,energy_order=24,phi_order=8,tail=70.)
 for k in['j','chi']:ck('full complex tensor absolute '+str((nu,el,c,k)),d[k]['log']-ref[k]['log'])
 row=dict(inputs=[nu,el,c],scalar=d,tensor={k:ref[k]for k in['j','chi']})
 if c<.99:
  p=O.Point(b.T,S.C.MN,S.C.MP,b.mu_n,b.mu_p,nu,el,c,S.C.ME,b.mu_e);v=O.integrate(p,order=32,nphi=8,tail=70.)
  for k in['j','chi']:ck('independent original delta spinor '+str((nu,el,c,k)),d[k]['value']/v[k]-1,3e-10)
  row['original_delta']={k:v[k]for k in['j','chi','max_shell_residual','order','nphi','tail']}
 points.append(row)
(P/'point_results.json').write_text(json.dumps(dict(checks=checks,points=points,passed=sum(v['passed']for v in checks),failed=[v for v in checks if not v['passed']]),indent=2)+'\n')
if not all(v['passed']for v in checks):raise RuntimeError('Absolute forward point gate failed, not integrating')
r=S.integrate(2.,lepton_order=6,angle_order=8,energy_order=8,phi_order=4,tail=40.,outer_tail=24.);(P/'profile.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(checks_passed=len(checks),pilot={k:r[k]for k in['j','chi','seconds','calls']}),indent=2))
