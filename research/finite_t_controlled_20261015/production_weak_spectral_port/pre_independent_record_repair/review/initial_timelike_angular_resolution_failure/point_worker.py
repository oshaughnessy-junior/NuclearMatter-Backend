from pathlib import Path
import sys,json,hashlib,math,time,importlib.util
import numpy as np
from scipy.integrate import quad
from scipy.special import expit,kve
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;s=importlib.util.spec_from_file_location('spectral_review_new_delta_reference',P/'reference.py');D=importlib.util.module_from_spec(s);sys.modules[s.name]=D;s.loader.exec_module(D);V=D.V;L=V.LOCK;A=json.loads((P/'AUTHOR_LOCK_ADMISSION.json').read_text());assert hashlib.sha256((P.parent/'production_weak_spectral_port/LOCK.json').read_bytes()).hexdigest()==A['author_lock_sha256'];checks=[];rows=[];start=time.monotonic();out=dict(checks=checks,rows=rows,stage='forwarddensity',candidate_imported=False)
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
def save():
 out.update(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],seconds=time.monotonic()-start);(P/'point_results.json').write_text(json.dumps(out,indent=2)+'\n')
def gateforward():
 save()
 if any(not c['passed']for c in checks):raise ArithmeticError('Forward/referencefailurepreserved')
state=json.loads((P.parent/'hot_ec_opacity_spectrum/results.json').read_text())['rows'][0]['rules'][2]['matched_state'];fd={}
for name,m,mu in[('neutron',D.MN,D.b.mu_n),('proton',D.MP,D.b.mu_p)]:
 scale=math.sqrt(2*m*D.b.T);n=quad(lambda y:y*y*expit((mu-math.hypot(m,scale*y))/D.b.T),0,math.sqrt(120),epsabs=1e-18,epsrel=3e-12)[0]*scale**3/(math.pi**2*D.HC**3);eta=(mu-m)/D.b.T;fug=m*m*D.b.T/(math.pi**2*D.HC**3)*sum((-1)**(k+1)*math.exp(k*eta)*kve(2,k*m/D.b.T)/k for k in range(1,6));fd[name]=dict(original_p=n,Bessel_fugacity=fug,saved=state[name]['density']);ck(name+' originalpVsBessel',n/fug-1,3e-9);ck(name+' originalpVssaveddensity',n/state[name]['density']-1,3e-9)
out['forwarddensities']=fd;gateforward();out['stage']='differential'
for point in L['differential_points']:
 ch=point['channel'];args=[ch,point['E'],point['El'],point['cosine']];ref=[]
 for profile in['differential_coarse','differential_fine']:ref.append(D.differential(*args,**L['profiles'][profile]))
 row=dict(input=point,delta=ref);rows.append(row)
 for key in['j','chi']:ck(point['label']+ch+key+' ownrefinement',math.expm1(ref[0]['log'+key]-ref[1]['log'+key]),3e-6)
 for r in ref:ck(point['label']+ch+' on_shell',r['shell'],3e-10)
 if ref[-1]['support']['kind'].startswith('spacelike'):
  sr=L['profiles']['spinor_reference'];o=V.O.integrate(V.O.Point(D.b.T,D.MN if ch=='EC'else D.MP,D.MP if ch=='EC'else D.MN,D.b.mu_n if ch=='EC'else D.b.mu_p,D.b.mu_p if ch=='EC'else D.b.mu_n,point['E'],point['El'],point['cosine'],D.ME,D.b.mu_e if ch=='EC'else-D.b.mu_e,ch=='PC',D.MW),order=sr['order'],nphi=sr['phi'],tail=sr['tail']);row['original_explicit_spinor']={k:o[k]for k in['j','chi','Emin','max_shell_residual','min_jacobian','order','nphi','tail']}
  for k in['j','chi']:ck(point['label']+ch+k+' absoluteoriginalspinor',math.expm1(ref[-1]['log'+k]-math.log(o[k])),A['preoutput_tightened_differential_literal_gate'])
 eq=D.b.mu_p+D.b.mu_e-D.b.mu_n;eq=eq if ch=='EC'else-eq;ck(point['label']+ch+' KMSafterdirectproducts',ref[-1]['logj']-ref[-1]['logchi']+(point['E']-eq)/D.b.T,3e-10);gateforward()
# Absolute pre-existing timelikepoint anchors newQrestintegration; no sourcecalls.
saved=json.loads((P.parent/'review_hot_ec_opacity_spectrum/results.json').read_text())['points'][-1];nu,el,c=saved['inputs'];r=D.differential('EC',nu,el,c,**L['profiles']['differential_fine']);out['saved_timelike_anchor']=r
for key in['j','chi']:ck('Qrest absoluteSAVEDtimelike '+key,math.expm1(r['log'+key]-math.log(saved['author'][key]['value'])),3e-10)
for rel,h in L['sourcepins'].items():ck('sourceimmutable '+rel,0 if hashlib.sha256((P.parent/rel).read_bytes()).hexdigest()==h else 1,0)
out['stage']='COMPLETE';gateforward();print('POINTS',out['passed'],'FAIL',len(out['failed']),'seconds',out['seconds'])
