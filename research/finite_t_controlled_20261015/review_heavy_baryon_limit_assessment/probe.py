"""Different review: Weyl-basis traces, Bessel roots and fresh phase profiles."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,time
import numpy as np
from scipy.special import kve,expit
from scipy.integrate import quad
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;R=P.parent;A=R/'heavy_baryon_limit_assessment';sys.dont_write_bytecode=True
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(A/'worker.py')=='f8e43c2918ae43fe10cefee267b09c8cfc42c8bd62a5e272334f790d218f669d'
assert sha(A/'MANIFEST.json')=='236dfb8dc548219467fc1dafe3ba1e69886e262a8f478b0d942b36958f2017ab'
before={str(f.relative_to(A)):sha(f)for f in A.rglob('*')if f.is_file()};lock=json.loads((A/'LOCK.json').read_text());checks=[];fresh=[];start=time.monotonic()
def ck(n,e,t):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
K=load('review_heavy_frozen_point_scalar',A/'point_current.py')
I=np.eye(2);Z=np.zeros((2,2));sig=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])];G=np.array([np.block([[Z,I],[I,Z]])]+[np.block([[Z,s],[-s,Z]])for s in sig]);G5=np.diag([-1,-1,1,1]);metric=np.array([1,-1,-1,-1]);left=G@(np.eye(4)-G5);V=G-K.GA*(G@G5)
def slash(p):return np.einsum('u,u,uab->ab',p,metric,G)
def trace_amp(k,pa,l,pb,ma,mb,anti):
 H=np.array([[np.trace((slash(pb)+mb*np.eye(4))@V[u]@(slash(pa)+ma*np.eye(4))@V[v])for v in range(4)]for u in range(4)])
 if anti:L=np.array([[np.trace(slash(k)@left[u]@(slash(l)-.51099895069*np.eye(4))@left[v])for v in range(4)]for u in range(4)])
 else:L=np.array([[np.trace((slash(l)+.51099895069*np.eye(4))@left[u]@slash(k)@left[v])for v in range(4)]for u in range(4)])
 return K.GF*K.GF*K.VUD*K.VUD/8*np.einsum('u,v,uv,uv->',metric,metric,L,H).real
mn0=939.56542052;mp0=938.27208816;M0=(mn0+mp0)/2;Q=mn0-mp0
# New original lab two-body shell, not author's CM test point construction.
for scale in[1.,8.]:
 mn=scale*M0+Q/2;mp=scale*M0-Q/2
 for anti,Enu,c in[(False,7.9,-.37),(True,19.6,.41)]:
  ma,mb=(mp,mn)if anti else(mn,mp);pa=np.array([ma,0,0,0]);k=np.array([Enu,0,0,Enu]);me=.51099895069
  def final(E):
   p=math.sqrt((E-me)*(E+me));l=np.array([E,p*math.sqrt(1-c*c),0,p*c]);return l,pa+k-l
  def shell(E):
   l,p=final(E);return p[0]*p[0]-sum(p[1:]**2)-mb*mb
  E=brentq(shell,math.nextafter(me,math.inf),Enu+ma-mb,xtol=1e-12);l,pb=final(E)
  actual=K.amplitude(k[None,:],pa[None,:],l[None,:],pb[None,:],ma,mb,anti)[0];ref=trace_amp(k,pa,l,pb,ma,mb,anti);ck('freshWeyltrace absolute '+str((scale,anti)),actual/ref-1,3e-10)
# Literal lambda1 trace anchor gates before any deformed total phase calls.
if any(not c['passed']for c in checks):raise ArithmeticError('Forward current anchor failed before limit review')
T,n,y=[lock['state'][k]for k in['T','nB','Ye']];HC=197.3269804
records={i:json.loads((A/f'lambda{i}_scalar_profile1.json').read_text())for i in[1,2,4,8]}
for scale,r in records.items():
 m=r['masses'];ck('Qphysicalfixed '+str(scale),m['actual_Q']-Q,2e-12);ck('mnconsistent '+str(scale),m['mn']-(scale*M0+Q/2),0);ck('mpconsistent '+str(scale),m['mp']-(scale*M0-Q/2),0)
 for ch,mass,mu,target in[('n',m['mn'],r['bath']['mu_n'],n*(1-y)),('p',m['mp'],r['bath']['mu_p'],n*y)]:
  eta=(mu-mass)/T;density=mass*mass*T/(math.pi**2*HC**3)*sum((-1)**(j+1)*math.exp(j*eta)*kve(2,j*mass/T)/j for j in range(1,17));ck('independentBesselforward '+str((scale,ch)),density/target-1,3e-10)
 ck('fixedelectronbath '+str(scale),r['bath']['mu_e']-records[1]['bath']['mu_e'],0)
 # Literal species energy accounting, no separate Q term.
 e=r['raw']['EC']['scenarios']['empty']['net'];p=r['raw']['PC']['sums'][0][0]
 ck('ECspeciesledger '+str(scale),(e['electron_energy']+e['proton_energy']-e['neutron_energy']-e['neutrino_energy'])/e['neutrino_energy'],3e-10)
 ck('PCspeciesledger '+str(scale),(p[4]+p[2]-p[3]-p[1])/p[1],3e-10)
# Shared qualified phase kernels, NEW quadrature profile, fresh FD roots each scale.
for scale in[1.,8.]:
 C=load('review_newheavy_EC_'+str(scale),R/'hot_electron_capture_moments/electron_capture.py');PC=load('review_newheavy_PC_'+str(scale),R/'review_pc_moments/reference.py');mn=scale*M0+Q/2;mp=scale*M0-Q/2;C.MN=mn;C.MP=mp;C.MW=scale*M0;PC.MN=mn;PC.MP=mp;PC.MW=scale*M0
 C.scalar_amplitude=lambda k,n,l,p:K.amplitude(k,n,l,p,mn,mp,False);PC.amplitude=lambda k,p,l,n:K.amplitude(k,p,l,n,mp,mn,True)
 b,st=C.matched(**lock['state']);ec=C.integrate(b,proton_order=24,electron_order=40,initial_order=10,outgoing_order=10,phi_order=10,tail=65.,scenarios=[dict(name='empty',mode='empty')]);pc=PC.integral(vars(b),order_n=30,order_e=48,order_angle=11,tail=70.);v=dict(EC=[ec['scenarios']['empty']['net']['number'],ec['scenarios']['empty']['net']['neutrino_energy']],PC=pc['sums'][0][0][:2]);fresh.append(dict(scale=scale,bath=vars(b),EOS=st,EC=ec,PC=pc,values=v));(P/'partial.json').write_text(json.dumps(dict(fresh=fresh,checks=checks),indent=2)+'\n')
 for ch in['EC','PC']:
  for k in range(2):ck('freshactualnewprofile '+str((scale,ch,k)),v[ch][k]/records[int(scale)]['values'][ch][k]-1,3e-6)
 ck('freshECtargetdensity '+str(scale),ec['proton_density']/(n*y)-1,3e-8);ck('freshPCtargetdensity '+str(scale),pc['neutron_density']/(n*(1-y))-1,3e-8)
# Independent charged-lepton rapidity integral, finite100T tail.
mu=records[1]['bath']['mu_e'];me=.51099895069;pref=K.GF**2*K.VUD**2*(1+3*K.GA**2)/(2*math.pi**3*6.582119569e-22);allowed={}
for ch in['EC','PC']:
 lo=math.acosh(Q/me)if ch=='EC'else 0.;hi=math.acosh((me+100*T+abs(mu))/me)
 def f(u,k):
  E=me*math.cosh(u);p=me*math.sinh(u);nu=E-Q if ch=='EC'else E+Q;occ=expit(((mu if ch=='EC'else-mu)-E)/T);return E*p*p*nu**(2+k)*occ
 allowed[ch]=[pref*(n*y if ch=='EC'else n*(1-y))*quad(lambda u:f(u,k),lo,hi,epsabs=1e-15,epsrel=2e-12)[0]for k in[0,1]]
errors={scale:{ch:[r['values'][ch][k]/allowed[ch][k]-1 for k in range(2)]for ch in['EC','PC']}for scale,r in records.items()}
for ch in['EC','PC']:
 for k in range(2):
  vals=[abs(errors[s][ch][k])for s in[1,2,4,8]]
  ck('lockedstrictmonotone '+str((ch,k)),0 if all(vals[i+1]<vals[i]for i in range(3))else 1,0)
  for i in range(3):ratio=vals[i+1]/vals[i];ck('lockederrorratio '+str((ch,k,i)),max(0.,.25-ratio,ratio-.75),0)
  ck('lockedlambda8onepercent '+str((ch,k)),vals[-1],.01)
for f,h in lock['pins'].items():ck('unchangedupstream '+f,0 if sha(R/f)==h else 1,0)
after={str(f.relative_to(A)):sha(f)for f in A.rglob('*')if f.is_file()};ck('wholefrozenauthorunchanged',0 if before==after else 1,0)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],fresh=fresh,allowed=allowed,errors=errors,seconds=time.monotonic()-start,source_unchanged=before==after,method='Weyl independenttrace/newlabpoints +16termBesselEOS +actualnewquadratureprofile insharedaccepted phasekernels; not independentphysicalmodel.')
(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'],out['seconds'])
