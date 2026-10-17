"""Prelocked fresh EC/PC detailed states and independent fit/allowed arithmetic."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'free_capture_moment_fit';sys.dont_write_bytecode=True
lock=json.loads((P/'VALIDATION_LOCK.json').read_text());locksha=hashlib.sha256((P/'VALIDATION_LOCK.json').read_bytes()).hexdigest()
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
F=load('review_frozen_free_capture_fit_API',A/'approximation.py');M=F.M;R=load('fresh_fit_independent_material_roots',ROOT/'review_transparent_balance_grid/probe.py');checks=[];rows=[];start=time.monotonic();before={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest()for q in A.rglob('*')if q.is_file()}
def ck(n,e,t=3e-6):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def profile(state,fine):
 T,n,y=state['T'],state['nB'],state['Ye'];b=R.eos(T,n,y);bath=M.EC.Bath(**b);ec=M.EC.integrate(bath,proton_order=28 if fine else 20,electron_order=48 if fine else 32,initial_order=12 if fine else 8,outgoing_order=12 if fine else 8,phi_order=12 if fine else 8,tail=60. if fine else 50.,scenarios=[dict(name='empty',mode='empty')]);pc=M.PC.integral(b,order_n=32 if fine else 28,order_e=52 if fine else 44,order_angle=12 if fine else 10,tail=60.);ev=ec['scenarios']['empty']['net'];pv=np.asarray(pc['sums'])[0,0];v=[ev['number'],ev['neutrino_energy'],float(pv[0]),float(pv[1])];return dict(bath=b,values=v,density=dict(EC=ec['proton_density']/(n*y)-1,PC=pc['neutron_density']/(n*(1-y))-1),raw=dict(EC=ec,PC=pc))
def independent_allowed(T,mu,ch):
 me=M.S.ME;Q=M.Q;lo=max(me,Q)if ch=='EC'else me;hi=max(lo,mu if ch=='EC'else-mu)+100*T;x,w=leggauss(128);a,b=math.acosh(lo/me),math.acosh(hi/me);u=a+(x+1)*(b-a)/2;e=me*np.cosh(u);p=me*np.sinh(u);en=e-Q if ch=='EC'else e+Q;f=expit((mu-e)/T)if ch=='EC'else expit((-mu-e)/T);base=w*(b-a)/2*e*p*p*en**2*f;return M.S.RATE_PREFACTOR*np.array([sum(base),sum(base*en)])
coef=json.loads((A/'COEFFICIENT_FREEZE.json').read_text());M.verify()
for n,h in lock['author_sources'].items():ck('preregistered identity '+n,0 if hashlib.sha256((A/n).read_bytes()).hexdigest()==h else 1,0)
old=json.loads((A/'LOCK.json').read_text());revised=json.loads((A/'REVISED_LOCK2.json').read_text());ck('same heldout states/basis/gate',0 if all(old[k]==revised[k]for k in ['heldout','basis','heldout_gate','source_refinement_gate'])else 1,0);ck('remove only rejected corner',0 if[x for x in old['training']if x not in revised['training']]==[dict(T=1.,nB=1e-6,Ye=.5)]else 1,0)
ck('freshpoints disjoint authored data',0 if all(s not in revised['training']+revised['heldout']for s in lock['states'])else 1,0)
for i,state in enumerate(lock['states']):
 if time.monotonic()-start>105:raise TimeoutError('Freshfitbatch105s prior to next pair')
 a=profile(state,False);b=profile(state,True);fit=F.approximate_moments(**state);pred=np.array([fit['moments'][k]for k in coef['outputs']]);d=np.array(b['values']);fraction=pred/d-1
 T,n,y=state['T'],state['nB'],state['Ye'];mu=b['bath']['mu_e'];eta=mu/T;bar=max((b['bath']['mu_n']-M.EC.MN)/T,(b['bath']['mu_p']-M.EC.MP)/T)
 ck('fresh actual lepton admission '+str(i),0 if abs(eta)<=3 else 1,0);ck('fresh actual baryon admission '+str(i),0 if bar<=-5 else 1,0)
 base=np.r_[n*y*independent_allowed(T,mu,'EC'),n*(1-y)*independent_allowed(T,mu,'PC')];basis=np.array([1,T,T*T,eta,eta*eta,T*eta,n/1e-6*T**-1.5]);manual=base*np.exp(basis@np.array(coef['coefficients']))
 for j,name in enumerate(coef['outputs']):
  ck(f'new{i} actual physical refinement {name}',a['values'][j]/d[j]-1);ck(f'new{i} independent baseline+fit {name}',pred[j]/manual[j]-1,3e-10);ck(f'new{i} HELDOUT1percent {name}',fraction[j],.01)
 for profile_name,z in [('coarse',a),('fine',b)]:
  for k,v in z['density'].items():ck(f'new{i} {profile_name} targetFD {k}',v,3e-8)
 row=dict(state=state,coarse=a,refined=b,approximation=fit,relative_errors=fraction.tolist(),actual_eta=eta,actual_baryon_eta=bar);rows.append(row);(P/'partial.json').write_text(json.dumps(dict(rows=rows,checks=checks),indent=2)+'\n')
# Independent QR solution from saved TRAINING only; never retune/freeze new model.
train=[json.loads((A/f'training_{i}.json').read_text())for i in range(11)];X=[];Y=[]
for i,r in enumerate(train):
 st=r['state'];T,n=st['T'],st['nB'];eta=r['refined']['bath']['mu_e']/T;X.append([1,T,T*T,eta,eta*eta,T*eta,n/1e-6*T**-1.5]);Y.append(np.log(np.asarray(r['refined']['values'])/r['allowed']));ck('training file pinned '+str(i),0 if hashlib.sha256((A/f'training_{i}.json').read_bytes()).hexdigest()==coef['training_files'][f'training_{i}.json']else 1,0)
X=np.asarray(X);Y=np.asarray(Y);q,r=np.linalg.qr(X);own=np.linalg.solve(r,q.T@Y);ck('training-only QR reproduces locked coefficients',np.max(abs(own-np.asarray(coef['coefficients']))),3e-12);ck('original rank7',np.linalg.matrix_rank(X)-7,0);ck('original empty heldout inventory',len(coef['heldout_files_before_freeze']),0)
qualification=json.loads((A/'HELDOUT_ASSESSMENT.json').read_text())
for i in range(8):
 raw=json.loads((A/f'heldout_{i}.json').read_text());v=np.array(raw['allowed'])*np.exp(np.array(raw['basis'])@np.array(coef['coefficients']));err=v/np.array(raw['refined']['values'])-1
 for j,k in enumerate(coef['outputs']):ck('saved heldout arithmetic '+str((i,k)),err[j]-qualification['rows'][i]['relative_errors'][j],3e-14);ck('saved individual heldout1percent '+str((i,k)),err[j],.01)
for n,h in json.loads((A/'SOURCE_PIN.json').read_text()).items():ck('physical source frozen '+n,0 if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h else 1,0)
after={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest()for q in A.rglob('*')if q.is_file()};ck('author files unchanged',int(before!=after),0);ck('validation lock unchanged',0 if hashlib.sha256((P/'VALIDATION_LOCK.json').read_bytes()).hexdigest()==locksha else 1,0)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,max_fresh_fractional_error=max(abs(e)for r in rows for e in r['relative_errors']),seconds=time.monotonic()-start,before=before,after=after);(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'],'max',out['max_fresh_fractional_error'],'seconds',out['seconds'])
