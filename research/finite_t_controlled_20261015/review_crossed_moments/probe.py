from pathlib import Path
import sys,json,time,math,hashlib,atexit
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'crossed_thermal_moments'));import moments as M
import reference as R
start=time.monotonic();checks=[];rows=[];partial=None;phase='start';files=[ROOT/'crossed_thermal_moments'/n for n in ['moments.py','results.json','spectrum_anchor.py','spectrum_results.json','initial_refinement_failure.json']]+[ROOT/'relativistic_crossed_contract/beta.py',ROOT/'muon_full_current/current.py'];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
def save():
 end={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
 (HERE/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks)and hashes==end,phase=phase,checks=checks,rows=rows,partial=partial,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256=end),indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item())+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def checkpoint(v):
 global partial
 partial=v;save()
 if time.monotonic()-start>108:raise TimeoutError('Bounded independent review budget')
T=.055;nB=2.3e-12;Ye=.31;b,material=M.K.matched_npe(T,nB,Ye);zeta=b.mu_n-b.mu_p-b.mu_e;muanti=zeta-.37*T;rows.append(dict(kind='inputs',T=T,nB=nB,Ye=Ye,muanti=muanti,material=material))
# Direct momentum FD density gates before rate tests, including pair electron charge.
for label,m,mu,target,pairs in [('n',R.MN,b.mu_n,nB*(1-Ye),False),('p',R.MP,b.mu_p,nB*Ye,False),('e',R.ME,b.mu_e,nB*Ye,True)]:
 hi=max(m,abs(mu))+80*T;pmax=math.sqrt((hi-m)*(hi+m));cuts=sorted(set([0.,pmax]+[math.sqrt((E-m)*(E+m))for E in [mu+k*T for k in[-12,-4,0,4,12]]if m<E<hi]));density=sum(quad(lambda p:p*p*(expit((mu-math.hypot(m,p))/T)-(expit((-mu-math.hypot(m,p))/T)if pairs else 0.))/(math.pi**2*R.HC**3),lo,hi,epsabs=1e-24,epsrel=2e-11)[0]for lo,hi in zip(cuts[:-1],cuts[1:]));gate(label+' independent physical FD density',density/target-1,3e-9)
phase='new independent total';refs=[]
for order,radial,ori,phi in [(20,8,8,8),(32,14,12,16)]:
 r=R.integrate(b,muanti,order,radial,ori,phi,checkpoint=checkpoint);refs.append(r);rows.append(dict(kind='independent',result=r));save()
phase='author fresh same physical state';scenarios=[dict(name='unblocked',mode='unblocked'),dict(name='empty',mode='empty'),dict(name='FD',mode='FD',mu_nu=muanti),dict(name='LTE',mode='FD',mu_nu=zeta)];author=M.integrate(b,rest_order=32,parent_order=20,orient_order=10,phi_order=16,scenarios=scenarios,budget_seconds=35);rows.append(dict(kind='author fresh',result=author));save()
for scenario in ['unblocked','empty','FD','LTE']:
 for direction,code in [('forward','F'),('backward','B')]:
  for i,name in enumerate(R.NAMES):
   a=refs[-1]['scenarios'][scenario][code][i];old=refs[0]['scenarios'][scenario][code][i];v=author['scenarios'][scenario][direction][name]['value']
   if a>0:
    gate(scenario+' '+direction+' independent refinement '+name,old/a-1,3e-6);gate(scenario+' '+direction+' independent total '+name,a/v-1,3e-6)
 r=refs[-1]['scenarios'][scenario];n=r['net'];scale=r['F'][1]+r['B'][1];gate(scenario+' net energy heavy subtraction',(n[3]+n[2]-n[4]+n[1])/scale,3e-9);gate(scenario+' direct energy separately accumulated',(n[5]-n[1])/scale,3e-9)
 if scenario in ['FD','LTE']:
  flux=r['F'][0]+r['B'][0];force=.37 if scenario=='FD'else 0;gate(scenario+' actual occupations chemical log',r['max_log_ratio_residual'],3e-9);gate(scenario+' entropy identity',(r['entropy']-n[0]*force)/flux,3e-9);gate(scenario+' nonnegative entropy',float(r['entropy']<0),0.)
  if scenario=='LTE':gate('new LTE net overgross',n[0]/flux,3e-9)
# Proper-time and Doppler anchors are not inferred from discrete ledgers.
r=refs[-1];u=r['scenarios']['unblocked']['F'];gate('unblocked true FD proper lifetime',u[0]/(r['proper_density']*r['rest_rate'])-1,3e-12)
for i,name in enumerate(['neutrino_energy','electron_energy','proton_energy'],1):gate('unblocked Doppler identity '+name,u[i]/(r['density']*r['rest_moments'][name])-1,3e-12)
gate('rest independent source/Dalitz absolute',r['rest_rate']/M.B.pair_integral(40)['rate']-1,3e-10)
gate('new FD density vs matchedn',r['density']/material['neutron']['density']-1,3e-9)
phase='complete';partial=None;save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
