"""Free p+e- <-> n+nue complete event moments, physical pair-rich FD bath.

Exact moving-target two-body phase space; no spectral opacity or full inventory.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass,asdict
import sys,importlib.util,hashlib,math,numbers,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit,logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
PINS={'relativistic_capture_oracle/events.py':'c2c4afc2797b812c0585cdf09c598553e45c275ff362b64277b31350d49c4a76','relativistic_free_eos/free_baryons.py':'aa5fc335de62dde0aa33e328410560741a55ce9dcd5fcbf8b3f4a625e5ffed5b','thermodynamics/lepton_reference.py':'f87ea4bcf8830174995bf9c7061a943052c2d3be8d4386befe3c90215ef97ec9'}
def verify_sources():
 for name,sha in PINS.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen physical source changed: '+name)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
verify_sources();E=load('electron_capture_pinned_events','relativistic_capture_oracle/events.py');EOS=load('electron_capture_pinned_eos','relativistic_free_eos/free_baryons.py')
MN=EOS.MN;MP=EOS.MP;ME=EOS.leptons.ELECTRON_MASS;MW=(MN+MP)/2;HC=E.HC;HBAR=E.HBAR
NAMES=('number','neutrino_energy','electron_energy','proton_energy','neutron_energy','material_energy_loss')
def finite_real(x):return isinstance(x,numbers.Real)and not isinstance(x,(bool,np.bool_))and math.isfinite(x)
@dataclass(frozen=True)
class Bath:
 T:float;mu_n:float;mu_p:float;mu_e:float
 def __post_init__(self):
  if not all(finite_real(v)for v in asdict(self).values())or self.T<=0:raise ValueError('Finite total chemical potentials and positive T required')
 @property
 def equilibrium_mu_nue(self):return self.mu_p+self.mu_e-self.mu_n
def matched(T=2.,nB=1e-7,Ye=.3):
 if not all(finite_real(v)for v in (T,nB,Ye))or T<=0 or nB<=0 or not 0<Ye<1:raise ValueError('Positive finite T,nB and interior Ye')
 n=EOS.state_density(T,nB*(1-Ye),MN);p=EOS.state_density(T,nB*Ye,MP);l=EOS.leptons.state_density(T,nB*Ye,ME)
 bath=Bath(T,n.chemical_potential,p.chemical_potential,l.chemical_potential)
 return bath,dict(T=T,nB=nB,Ye=Ye,neutron=n.asdict(),proton=p.asdict(),electron=asdict(l),muons='absent')
def represented(log):
 if log is None:return dict(value=0.,log=None,representable=True,reason='absent_initial_nue_occupation')
 if not math.isfinite(log):raise ArithmeticError('Finite log moment required')
 try:v=math.exp(log)
 except OverflowError:return dict(value=None,log=log,representable=False,reason='overflow')
 return dict(value=v if v else None,log=log,representable=v>0,reason=None if v else 'underflow')
def dot(a,b):return a[:,0]*b[:,0]-np.sum(a[:,1:]*b[:,1:],axis=1)
def spinor_amplitude(k,n,l,p):
 """Literal reciprocal nbar adjΓ u_p * nubar γ(1-γ5)u_e, initial spin avg4."""
 un=E.spinors(n,MN);up=E.spinors(p,MP);uk=E.spinors(k,0);ul=E.spinors(l,ME)
 V=E.adj(E.vertices(p-n,MW));left=E.G@(E.I-E.G5)
 J=np.einsum('nai,ab,nubc,ncj->nuij',un.conj(),E.G[0],V,up).reshape((-1,4,4))
 L=np.einsum('nai,ab,ubc,ncj->nuij',uk.conj(),E.G[0],left,ul).reshape((-1,4,4))
 amp=np.einsum('nui,u,nuj->nij',L,E.MET,J)
 return E.GF**2*E.VUD**2/8*np.sum(abs(amp)**2,axis=(1,2))
def scalar_amplitude(k,n,l,p):
 """Guo47 invariants for reciprocal ν+n capture, avg4 for original p+e-.

    Adapted from accepted review_pc_moments polynomial with physical n->p
    ordering and NU (not anti) VA/AF signs. Checked against literal spinors.
 """
 q=p-n;t=dot(q,q)
 if np.any(t>=min(139.57**2,840.**2,1000.**2,4*MW*MW)):raise ValueError('Subpole free-current events required')
 den=(1-t/(4*MW*MW))*(1-t/840**2)**2;v=(1-4.706*t/(4*MW*MW))/den;a=1.2723/(1-t/1000**2)**2;f=3.706/den;g=2*MW*MW*a/(139.57**2-t)
 kn,kp,ln,lp,kl,np_=[dot(x,y)for x,y in[(k,n),(k,p),(l,n),(l,p),(k,l),(n,p)]];kq,lq,nq,pq=[dot(x,q)for x in[k,l,n,p]];ma=MN;mb=MP
 VV=v*v*(kn*lp+kp*ln-ma*mb*kl);VA=2*v*a*(kn*lp-kp*ln);AA=a*a*(kn*lp+kp*ln+ma*mb*kl)
 VF=v*f/(2*MW)*((kn*mb-kp*ma)*lq+(ln*mb-lp*ma)*kq+(nq*mb-pq*ma)*kl)
 AF=a*f/MW*((kn*mb+kp*ma)*lq-(ln*mb+lp*ma)*kq)
 FF=f*f/(8*MW*MW)*(2*(ln*pq+lp*nq)*kq+2*(kn*pq+kp*nq)*lq-2*kq*lq*np_+t*(kl*np_-2*kn*lp-2*kp*ln)-ma*mb*(kl*t+2*kq*lq))
 AP=a*g/MW*(kq*(ln*mb-lp*ma)+lq*(kn*mb-kp*ma)+kl*(pq*ma-nq*mb))
 PP=g*g/(2*MW*MW)*(2*kq*lq-t*kl)*(np_-ma*mb)
 return 8*E.GF**2*E.VUD**2*(VV+VA+AA+VF+AF+FF+AP+PP)
def threshold(Ep,P,c):
 """Exact incident electron TOTAL-energy threshold at fixed proton and cosine."""
 D=(MN*MN-MP*MP-ME*ME)/2;A=Ep*Ep-P*P*c*c
 if Ep*ME>=D or np.any(A<=0):raise ValueError('Selected simple threshold branch requires Ep*me < D')
 return (Ep*D+P*c*np.sqrt(D*D-A*ME*ME))/A
def mapped(order,T,upper):
 x,w=leggauss(order);y=(x+1)*math.sqrt(upper/T)/2
 return T*y*y,w*math.sqrt(upper/T)*T*y
def integrate(bath,*,proton_order=20,electron_order=32,initial_order=8,outgoing_order=8,phi_order=8,tail=50.,scenarios=None,amplitude='scalar',budget_seconds=110.):
 verify_sources()
 if not isinstance(bath,Bath):raise ValueError('Explicit matched physical free npe Bath required')
 if any(isinstance(z,bool)or not isinstance(z,int)or z<4 for z in [proton_order,electron_order,initial_order,outgoing_order,phi_order]):raise ValueError('Integer orders>=4')
 if not finite_real(tail)or not 20<=tail<=100 or not finite_real(budget_seconds)or not 0<budget_seconds<=120 or amplitude not in ['scalar','spinor']:raise ValueError('Finite tail20..100,budget<=120 and named amplitude required')
 if not .2<=bath.T<=4 or max((bath.mu_n-MN)/bath.T,(bath.mu_p-MP)/bath.T)>-5 or abs(bath.mu_e)/bath.T>3:raise ValueError('Selected pair-rich T.2..4,dilute baryon eta<=-5,|mue/T|<=3 only')
 if scenarios is None:scenarios=[dict(name='empty',mode='empty'),dict(name='LTE',mode='FD',mu_nu=bath.equilibrium_mu_nue),dict(name='noneq',mode='FD',mu_nu=bath.equilibrium_mu_nue-.6*bath.T)]
 if not scenarios or len({s['name']for s in scenarios})!=len(scenarios):raise ValueError('Nonempty unique named scenarios')
 for s in scenarios:
  if s['mode']not in ['empty','FD']or any(k not in ['name','mode','mu_nu','temperature']for k in s):raise ValueError('Explicit empty/common-T FD radiation only')
  if s['mode']=='FD'and(not finite_real(s.get('mu_nu'))or abs(s['mu_nu']/bath.T)>10):raise ValueError('Finite common-T FD potential |mu/T|<=10')
  if 'temperature'in s and s['temperature']!=bath.T:raise ValueError('Radiation temperature must explicitly equal matter T')
  if s['mode']=='empty'and 'mu_nu'in s:raise ValueError('Empty radiation has no FD chemical potential')
 mu_rad=max([s.get('mu_nu',0.)for s in scenarios]+[0.]);kpmax=max(0.,bath.mu_p-MP,bath.mu_n+mu_rad-MP-ME)+tail*bath.T
 emax=max(MN-MP,ME,bath.mu_e,bath.mu_n+mu_rad-MP)+tail*bath.T
 pk,pw=mapped(proton_order,bath.T,kpmax);ci,wi=leggauss(initial_order);co,wo=leggauss(outgoing_order);ph=2*math.pi*np.arange(phi_order)/phi_order;xe,we=leggauss(electron_order)
 acc={s['name']:dict(F={n:[]for n in NAMES},B={n:[]for n in NAMES},net=np.zeros(6),entropy=0.,kms=0.)for s in scenarios}
 start=time.monotonic();density=0.;maxfour=maxmass=maxledger=maxth=0.;amin=math.inf;points=0
 for kin,wkin in zip(pk,pw):
  if time.monotonic()-start>budget_seconds:raise TimeoutError('Bounded actual electron-capture total; no partial qualified')
  Ep=MP+kin;P=math.sqrt(kin*(kin+2*MP));eth=threshold(Ep,P,ci)
  if np.any(eth>=emax):raise ArithmeticError('Upper tail does not enclose all initial thresholds')
  # Electron quadrature starts at the exact moving-target/angle threshold.
  ymax=np.sqrt((emax-eth)/bath.T);y=(xe[:,None]+1)*ymax[None,:]/2;el=eth[None,:]+bath.T*y*y;jac=we[:,None]*ymax[None,:]*bath.T*y
  EL,CI,CO,PH=np.meshgrid(np.arange(electron_order),np.arange(initial_order),co,ph,indexing='ij');ii=EL.astype(int).ravel();jj=CI.astype(int).ravel();ee=el[ii,jj];c=ci[jj];cc=CO.ravel();phi=PH.ravel();pe=np.sqrt((ee-ME)*(ee+ME))
  weights=jac[ii,jj]*wi[jj]*np.tile(np.repeat(wo,phi_order),electron_order*initial_order)*2*math.pi/phi_order
  N=len(ee);p=np.tile([Ep,0,0,P],(N,1));l=np.column_stack([ee,pe*np.sqrt(1-c*c),np.zeros(N),pe*c]);total=p+l;s=dot(total,total);root=np.sqrt(s);beta=total[:,1:]/total[:,0,None]
  if np.any(s<=MN*MN):raise ArithmeticError('Threshold-resolved events must be strictly physical')
  pf=(s-MN*MN)/(2*root);lcm=E.boost(l,-beta);axis=lcm[:,1:]/np.linalg.norm(lcm[:,1:],axis=1)[:,None];ey=np.tile([0.,1.,0.],(N,1));ex=np.cross(ey,axis)
  direction=cc[:,None]*axis+np.sqrt(1-cc*cc)[:,None]*(np.cos(phi)[:,None]*ex+np.sin(phi)[:,None]*ey)
  k=E.boost(np.column_stack([pf,pf[:,None]*direction]),beta);n=E.boost(np.column_stack([(s+MN*MN)/(2*root),-pf[:,None]*direction]),beta)
  amp=(scalar_amplitude if amplitude=='scalar'else spinor_amplitude)(k,n,l,p);amin=min(amin,float(min(amp)))
  # spin2 proton radial, spin2 electron relative-angle phase, avg4 amplitude.
  # Moller flux and initial CM momentum cancel exactly against dσ/dΩ.
  weight=P*Ep*wkin/math.pi**2 *pe*ee/(2*math.pi**2)*weights*amp*pf/(64*math.pi**2*root*Ep*ee*HBAR*HC**3)
  if np.any(weight<=0)or not np.all(np.isfinite(weight)):raise ArithmeticError('Positive physical event weights unresolved')
  fp=float(log_expit((bath.mu_p-Ep)/bath.T));nfp=float(log_expit((Ep-bath.mu_p)/bath.T));fl=log_expit((bath.mu_e-ee)/bath.T);nfl=log_expit((ee-bath.mu_e)/bath.T);fn=log_expit((bath.mu_n-n[:,0])/bath.T);nfn=log_expit((n[:,0]-bath.mu_n)/bath.T)
  density+=P*Ep*wkin/math.pi**2/HC**3*math.exp(fp);loss=Ep+ee-n[:,0];values=np.column_stack([np.ones(N),k[:,0],ee,np.full(N,Ep),n[:,0],loss]);base=np.log(weight)
  if np.any(values<=0):raise ArithmeticError('Positive event energies/direct escaped loss unresolved')
  maxfour=max(maxfour,float(np.max(abs(p+l-n-k))));maxmass=max(maxmass,float(np.max(abs(dot(n,n)-MN*MN)))/(MN*MN));maxledger=max(maxledger,float(np.max(abs(loss-k[:,0]))));points+=N
  DD=(MN*MN-MP*MP-ME*ME)/2;maxth=max(maxth,float(max(abs(Ep*eth-P*np.sqrt(eth*eth-ME*ME)*ci-DD)))/DD)
  for scenario in scenarios:
   a=acc[scenario['name']]
   if scenario['mode']=='empty':lf=fp+fl+nfn;lb=None
   else:
    fr=log_expit((scenario['mu_nu']-k[:,0])/bath.T);nfr=log_expit((k[:,0]-scenario['mu_nu'])/bath.T);lf=fp+fl+nfn+nfr;lb=nfp+nfl+fn+fr;delta=lf-lb;expected=(bath.equilibrium_mu_nue-scenario['mu_nu'])/bath.T;a['kms']=max(a['kms'],float(max(abs(delta-expected))))
   for i,name in enumerate(NAMES):
    a['F'][name].append(float(logsumexp(base+lf+np.log(values[:,i]))))
    if lb is not None:a['B'][name].append(float(logsumexp(base+lb+np.log(values[:,i]))))
   if lb is None:net=np.exp(base+lf)
   else:
    net=np.sign(delta)*np.exp(base+np.maximum(lf,lb))*(-np.expm1(-abs(delta)));a['entropy']+=float(net@delta)
   a['net']+=net@values
 out={}
 for scenario in scenarios:
  a=acc[scenario['name']];F={k:represented(float(logsumexp(v)))for k,v in a['F'].items()};B={k:represented(float(logsumexp(v))if v else None)for k,v in a['B'].items()};net=dict(zip(NAMES,a['net'].tolist()));scale=F['neutrino_energy']['value']+(B['neutrino_energy']['value']or 0.);material=net['neutron_energy']-net['proton_energy']-net['electron_energy']
  out[scenario['name']]=dict(scenario=scenario,forward=F,backward=B,net=net,material_energy_change=material,direct_material_energy_change=-net['material_energy_loss'],radiation_energy_change=net['neutrino_energy'],energy_ledger_scaled=(material+net['neutrino_energy'])/scale,direct_energy_ledger_scaled=(net['neutrino_energy']-net['material_energy_loss'])/scale,number_sources=dict(n=net['number'],p=-net['number'],electron_net=-net['number'],nue=net['number']),matter_Le_change=-net['number'],radiation_Le_change=net['number'],entropy=a['entropy']if scenario['mode']=='FD'else None,max_event_kms_error=a['kms'])
 return dict(bath=asdict(bath),scenarios=out,proton_density=density,controls=dict(proton_order=proton_order,electron_order=electron_order,initial_order=initial_order,outgoing_order=outgoing_order,phi_order=phi_order,tail=tail,proton_kinetic_max=kpmax,electron_total_max=emax,amplitude=amplitude),points=points,min_amplitude=amin,max_four_residual=maxfour,max_neutron_mass_residual=maxmass,max_single_event_energy_residual=maxledger,max_threshold_shell_residual=maxth,seconds=time.monotonic()-start,units='number fm^-3s^-1; energy MeVfm^-3s^-1; entropy kB fm^-3s^-1',inventory='p+e- <-> n+nue ONLY, physical free masses; no beta/PC join or full inventory',source_pins=PINS)
