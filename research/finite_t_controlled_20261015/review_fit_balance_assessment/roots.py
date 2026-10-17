import time,json,math
from common import *
from scipy.optimize import brentq
start=time.monotonic();before=inventory();checks=[];rows=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(math.isfinite(error)and abs(error)<=gate)))
for i,ar in enumerate(AUTHOR['rows']):
 T,n,y0=ar['T'],ar['nB'],ar['candidate_root'];trials=[]
 def trial(y):
  if time.monotonic()-start>105:raise TimeoutError('review root batch105s')
  verify();bath,s=eos(T,n,y);fit=F.approximate_moments(T,n,float(y));bb=BETA.integrate(BETA.K.Bath(**bath),rest_order=20,parent_order=26,orient_order=10,phi_order=14,coordinate='pair',parent_coordinate='direct_p',tail=75.,scenarios=[dict(name='empty',mode='empty')])
  beta=bb['scenarios']['empty']['net'];v=fit['moments'];r=beta['number']+v['R_PC']-v['R_EC'];gross=beta['number']+v['R_PC']+v['R_EC']
  ck(f'{i} beta actual FD parent {len(trials)}',bb['parent_density']/(n*(1-y))-1,3e-8)
  ck(f'{i} neutral EOS {len(trials)}',s['electron']['density']/(n*y)-1,1e-10)
  ck(f'{i} fit fresh EOS mu {len(trials)}',fit['state']['mu_e']-bath['mu_e'],3e-10)
  trials.append(dict(Ye=y,bath=bath,eos=s,fit=fit,beta=beta,R=r,gross=gross))
  return r
 left,right=trial(y0-.0008),trial(y0+.0008)
 ck(f'{i} left sign',0 if left>0 else 1,0);ck(f'{i} right sign',0 if right<0 else 1,0)
 y=brentq(trial,y0-.0008,y0+.0008,xtol=3e-13,rtol=1e-13);trial(y);state=trials[-1];bath,s=state['bath'],state['eos'];direct,accepted=C.matched(T,n,y)
 ck(f'{i} new approximate root',y-y0,3e-9);ck(f'{i} approximate numerical residual',state['R']/state['gross'],1e-9)
 for field in ['mu_n','mu_p','mu_e']:ck(f'{i} independent EOS {field}',bath[field]-getattr(direct,field),3e-10)
 for species,target in [('neutron',n*(1-y)),('proton',n*y)]:ck(f'{i} Bessel vs original momentum {species}',s[species]['density']/target-1,1e-10)
 # Independent fixed-T composition derivative, at two symmetric h values.
 finite=[]
 for h in [3e-4,1.5e-4]:
  derivative=(eos(T,n,y+h)[1]['energy']-eos(T,n,y-h)[1]['energy'])/(2*h*n);finite.append(derivative);ck(f'{i} directional uY h{h}',derivative-s['uY'],2e-5)
 ck(f'{i} uY versus accepted state response',s['uY']-ar['fit_beta_candidate']['uY_mev'],2e-5)
 fine=electron(T,bath['mu_e'],112)
 for key in ['density','energy','epsilon','chi']:ck(f'{i} electron momentum order {key}',fine[key]/s['electron'][key]-1,3e-11)
 ck(f'{i} locked model root shift',y-ar['reference_root'],.001)
 rows.append(dict(index=i,T=T,nB=n,root=y,author_root=y0,reference_root=ar['reference_root'],left_R=left,right_R=right,trials=trials,finite_uY=finite))
for x in LOCK['outside_domain']:
 try:F.approximate_moments(**x);rejected=False
 except ValueError:rejected=True
 ck('outside retained Ye rejects '+str(x),0 if rejected else 1,0)
after=inventory();ck('all frozen input bytes unchanged',0 if before==after else 1,0)
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,rows=rows,before=before,after=after,seconds=time.monotonic()-start)
(P/'roots_results.json').write_text(json.dumps(out,indent=2)+'\n');print(dict(passed=out['passed'],failed=out['failed'],roots=[r['root']for r in rows],seconds=out['seconds']))
assert not out['failed']
