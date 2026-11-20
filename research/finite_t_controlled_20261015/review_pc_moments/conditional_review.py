"""Frozen cone phase ordering, independent invariant amplitude + guard audit.

Function clone redirects only an in-memory primitive namespace; author
modules and files are unchanged. This is NOT an independent cone quadrature.
"""
from pathlib import Path
import sys,json,math,time,types,hashlib
import numpy as np
sys.dont_write_bytecode=True
from reference import *
H=Path(__file__).parent;R=H.parent;A=R/'positron_capture_moments';sys.path.insert(0,str(A));import moments as M;import conditional as C
start=time.monotonic();initial={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in list(A.iterdir())+[R/'positron_capture/refinement.json',R/'relativistic_capture_oracle/events.py']if p.is_file()};checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
s=json.loads((A/'MATCHED_STATE.json').read_text());b=M.PC.crossed.Bath(s['T'],s['neutron']['chemical_potential'],s['proton']['chemical_potential'],s['electron']['chemical_potential'])
proxyE=types.SimpleNamespace(**vars(M.E));proxyE.amplitude=lambda eb,k,p,l,n:2*amplitude(k,p,l,n)
proxyM=types.SimpleNamespace(**vars(M));proxyM.E=proxyE
clone=types.FunctionType(C.spectrum.__code__,dict(C.spectrum.__globals__,M=proxyM),'cone_invariant',C.spectrum.__defaults__)
reference=clone(b,5.,norder=24,corder=14,eorder=20,phi_order=10,tail=50.)
author=next(r['result']for r in reversed(json.loads((A/'refinement.json').read_text())['rows'])if r['kind']=='conditional')
for key,target in [('j',.3302220190349698),('chi',.987407803020526)]:
 gate('independent amplitude cone '+key,reference[key]['value']/author[key]['value']-1,3e-6)
 gate('selected absolute response '+key,reference[key]['value']/target-1,3e-6)
# Verify accepted response target is actually present in pinned data, not an
# unproven number carried in commentary.
text=(R/'positron_capture/refinement.json').read_text();parsed=json.loads(text)
def leaves(x):
 if isinstance(x,dict):
  for a in x.values():yield from leaves(a)
 elif isinstance(x,list):
  for a in x:yield from leaves(a)
 elif isinstance(x,(int,float)):yield x
values=list(leaves(parsed))
for key,target in [('j',.3302220190349698),('chi',.987407803020526)]:gate('pinned source target '+key,float(not any(abs(v/target-1)<1e-13 for v in values if v)),0.)
def records(x):
 if isinstance(x,dict):
  yield x
  for v in x.values():yield from records(v)
 elif isinstance(x,list):
  for v in x:yield from records(v)
matched_record=next(x for x in records(parsed)if isinstance(x.get('j'),dict)and x['j'].get('value')and abs(x['j']['value']/.3302220190349698-1)<1e-13)
gate('accepted response same actual bath',float(matched_record['bath']!=vars(b)),0.)
gate('accepted response same energy',matched_record['Enu']-5.,0.)
dos=5.**2/(2*math.pi**2*HC**3);gate('one helicity DOS',reference['DOS']/dos-1,1e-14)
# Independent cone Jacobian from direct Lorentz E transform at a fresh event.
En=MN+3.1;pn=math.sqrt(3.1*(3.1+2*MN));El=4.2;pe=math.sqrt(El*El-ME*ME);ci=.37;P=np.array([En+El,pe*math.sqrt(1-ci*ci),0.,pn+pe*ci]);s2=P[0]**2-np.dot(P[1:],P[1:]);rt=math.sqrt(s2);speed=np.linalg.norm(P[1:])/P[0];gamma=P[0]/rt;pf=(s2-MP*MP)/(2*rt);co=.31;h=1e-5
fd=(gamma*pf*(1+speed*(co+h))-gamma*pf*(1+speed*(co-h)))/(2*h)
gate('independent boost cone delta Jacobian',fd/(gamma*speed*pf)-1,2e-9)
rows.append(dict(conditional_reference=reference,author_comparison=author,independent_cone_jacobian=dict(inverse=1/fd,exact_inverse=1/(gamma*speed*pf))))
for name,fn in [('empty scenarios',lambda:M.integrate(b,scenarios=[])),('NaN tail',lambda:M.integrate(b,tail=math.nan)),('NaN radiation',lambda:M.integrate(b,scenarios=[dict(name='x',mode='FD',mu_nu=math.nan)])),('budget121',lambda:M.integrate(b,budget_seconds=121)),('boolean radial',lambda:M.integrate(b,neutron_order=True)),('conditional Ezero',lambda:C.spectrum(b,0)),('conditional orderbool',lambda:C.spectrum(b,5,norder=True))]:
 try:fn();rejected=False
 except ValueError:rejected=True
 gate('reject '+name,float(not rejected),0.)
# Initial coarse failures and both integration stops remain excluded.
coarse=json.loads((A/'results.json').read_text());coarse_checks=coarse.get('checks',[]);rows.append(dict(preserved_coarse_counts=dict(total=len(coarse_checks),failed=sum(not c['passed']for c in coarse_checks)),runtime_exclusion=(A/'initial_refinement_timeout.txt').read_text()))
for path,sha in initial.items():gate('source bytes '+path,float(hashlib.sha256((R/path).read_bytes()).hexdigest()!=sha),0.)
out=dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,source_hashes=initial,runtime_seconds=time.monotonic()-start)
(H/'conditional_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[c for c in checks if not c['passed']],runtime=out['runtime_seconds'])))
