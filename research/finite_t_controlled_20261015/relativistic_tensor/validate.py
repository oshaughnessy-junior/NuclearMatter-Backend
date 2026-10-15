"""Bounded differential-tensor author evidence; no integrated opacity claim."""
from pathlib import Path
import math,json,time,hashlib
import numpy as np
import tensor as T
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
def convert(x):
 if isinstance(x,np.ndarray):return dict(real=x.real.tolist(),imag=x.imag.tolist())if np.iscomplexobj(x)else x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
cases=[dict(label='electron',state=T.State(4.,T.V.MN,T.V.MP,945.,938.),Enu=40.,El=38.,cosine=.2,lepton_mass=T.V.ME,mu_lepton=20.),
 dict(label='muon',state=T.State(12.,T.V.MN,T.V.MP,950.,940.),Enu=150.,El=120.,cosine=-.3,lepton_mass=T.V.MMU,mu_lepton=80.),
 dict(label='antimuon',state=T.State(12.,T.V.MP,T.V.MN,940.,950.),Enu=160.,El=125.,cosine=.4,lepton_mass=T.V.MMU,mu_lepton=-30.,antineutrino=True),
 dict(label='timelikeelectron',state=T.State(4.,T.V.MN,T.V.MP,945.,938.),Enu=.05,El=1.25,cosine=.8,lepton_mass=T.V.ME,mu_lepton=20.)]
for case in cases:
 case=case.copy();label=case.pop('label');state=case.pop('state');run=[]
 for order,phi,tail in[(16,8,45.),(24,8,60.),(32,16,80.)]:
  if time.monotonic()-start>105:raise TimeoutError('Tensor105s budget')
  run.append(T.differential(state,**case,energy_order=order,phi_order=phi,tail=tail))
 row=dict(label=label,state=T.asdict(state),inputs=case,runs=run);rows.append(row)
 for i,r in enumerate(run):
  zeta=case['mu_lepton']+state.mu_b-state.mu_a
  gate(f'{label}{i} KMS',r['j']['log']-r['chi']['log']+(case['Enu']-zeta)/state.T,3e-10)
  for key in('absorption','emission'):
   W=r['response'][key]['scaled'];norm=np.max(abs(W))
   gate(f'{label}{i}{key} Hermiticity',np.max(abs(W-W.conjugate().T))/norm,3e-13)
   gate(f'{label}{i}{key} positive semidefinite',min(float(np.linalg.eigvalsh(W)[0])/norm,0.),3e-12)
  gate(f'{label}{i} positive coefficients',0. if r['j']['value']>0 and r['chi']['value']>0 else 1.,0.)
 for i in(0,1):
  for key in('j','chi'):gate(f'{label}{i} refinement {key}',math.expm1(run[i][key]['log']-run[2][key]['log']),3e-6)
  for key in('absorption','emission'):
   x,y=run[i]['response'][key],run[2]['response'][key]
   a=x['scaled']*math.exp(x['log_scale']-y['log_scale']);b=y['scaled']
   gate(f'{label}{i} allW refinement {key}',np.max(abs(a-b))/np.max(abs(b)),3e-6)

# Exact support algebra: compare returned intervals against original angle test.
support_cases=[(939.,938.,2.,50.),(1000.,900.,-90.,30.),(900.,1000.,90.,30.),(T.V.MN,T.V.MP,-.5,.5),(T.V.MN,T.V.MP,.5,.5),(939.,939.,200.,50.)]
support_rows=[]
for ma,mb,w,q in support_cases:
 intervals=T.support(ma,mb,w,q);support_rows.append(dict(ma=ma,mb=mb,omega=w,q=q,intervals=intervals))
 for Ea in np.linspace(max(ma,mb-w)+.01,5000.,51):
  p=math.sqrt((Ea-ma)*(Ea+ma));x=((ma-mb)*(ma+mb)+w*w-q*q+2*w*Ea)/(2*p*q)
  direct=abs(x)<=1 and Ea+w>=mb
  predicted=any(Ea>=r['lower']and(r['upper']is None or Ea<=r['upper'])for r in intervals)
  gate(f'support direct ma{ma}w{w}E{Ea}',0. if direct==predicted else 1.,0.)
linear_state=T.State(20.,T.V.MN,T.V.MP,950.,940.)
linear=[T.response(linear_state,-.5,.5,energy_order=o,phi_order=ph,tail=tail)for o,ph,tail in[(16,8,50.),(32,16,80.)]]
for key in('absorption','emission'):
 a,b=linear[0][key],linear[1][key]
 gate('actual lightlike tensor refinement '+key,np.max(abs(a['scaled']*math.exp(a['log_scale']-b['log_scale'])-b['scaled']))/np.max(abs(b['scaled'])),3e-6)
 gate('actual lightlike nonempty '+key,0. if b['log_scale']is not None and np.max(abs(b['scaled']))>0 else 1.,0.)

# Matched common-mass NR limit: density-only spinor tensor vs frozen S_NR.
sys=T.sys;sys.path.insert(0,str(T.ROOT/'response_reference'));from free_density_response import PairState,log_response
for M in(1e4,1e5):
 state=T.State(2.,M,M,M+3.,M+1.,weak_mass=939.)
 model=T.C.FormFactors(gv=1.,ga=0.,weak_magnetism=False,pseudoscalar=False,q2_dependence=False)
 omega=20*math.sqrt(2./M)*.3+200/M
 r=T.response(state,omega,20.,model=model,energy_order=32,phi_order=8,tail=80.)
 density=r['absorption']['scaled'][0,0].real;logdensity=math.log(density)+r['absorption']['log_scale']
 expected=log_response(PairState(2.,M,0.,0.,3.,1.),20.,omega)+3*math.log(T.V.HBARC)
 gate(f'NR density limit M{M}',math.expm1(logdensity-expected),3e-3 if M==1e4 else 3e-4)
sys.path.insert(0,str(T.ROOT/'allowed_opacity'));import allowed_capture as NR
for M in(1e5,1e6):
 state=T.State(2.,M,M,M+3.,M+1.,weak_mass=939.)
 model=T.C.FormFactors(weak_magnetism=False,pseudoscalar=False,q2_dependence=False)
 r=T.differential(state,10.,9.995,.2,T.V.ME,2.,model=model,energy_order=32,phi_order=8,tail=80.)
 pair=PairState(2.,M,0.,0.,3.,1.);q=r['response']['q'];w=.005;El=9.995;pl=math.sqrt(El*El-T.V.ME**2)
 common=math.log(NR.K*El*pl*NR.angular_factor(El,.2))
 chi=common+log_response(pair,q,w)+float(T.log_expit((El-2.)/2.))
 j=common+log_response(pair.reverse(),q,-w)+float(T.log_expit((2.-El)/2.))
 for key,expected in [('chi',chi),('j',j)]:gate(f'absolute NR allowed differential M{M}{key}',math.expm1(r[key]['log']-expected),3e-4 if M==1e5 else 3e-5)

# Same bare weak normalization even when spinor masses are changed.
state=T.State(8.,700.,710.,710.,715.,weak_mass=(T.V.MN+T.V.MP)/2)
r=T.response(state,-5.,80.,energy_order=24,phi_order=8,tail=60.)
expected=T.C.FormFactors().values(25-6400,state.weak_mass)
gate('independent bare Mweak normalization',max(abs(a-b)for a,b in zip(r['form_factors'],expected)),0.)
gate('combined representation avoids premature scale underflow',T._representation(np.eye(4,dtype=complex)*1e10,-750.)['tensor'][0,0].real/math.exp(math.log(1e10)-750.)-1,3e-14)

out=dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,support_cases=support_rows,lightlike_response=linear,runtime_seconds=time.monotonic()-start,scope='selected differential free particle-hole tensors; no lepton-integrated opacity or GM1',source_sha256={str(p.relative_to(T.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),P/'tensor.py',T.ROOT/'muon_full_current/current.py',T.ROOT/'muon_vacuum/vacuum_cc.py']})
(P/'results.json').write_text(json.dumps(out,default=convert,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],checks=len(checks),seconds=out['runtime_seconds'],failed=[x for x in checks if not x['passed']])))
if not out['passed']:raise SystemExit(1)
