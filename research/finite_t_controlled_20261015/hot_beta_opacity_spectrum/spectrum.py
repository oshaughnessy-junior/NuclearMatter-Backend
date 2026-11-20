"""Actual fixed-bath beta/inverse3body spectral j/truechi by Doppler cone delta.

Shared accepted full rest-current primitive; no KMS-generated inverse.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util,sys,math,time,json,hashlib,numbers
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit,logsumexp
R=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
HOT_RESULTS_SHA='57e2e89e820b0e80b9d0cd4956d67c54e208cd78d5e2609edcd8e1ef54f94d88'
PINS=json.loads((R/'hot_beta_moments/CHECKPOINT.json').read_text())['upstream'];PINS['hot_beta_moments/results.json']=HOT_RESULTS_SHA;PINS['hot_beta_moments/hot_moments.py']='73ae01ded717d58536c1d816152bae0b19d3bf425bd2a648c711b5b9942e049a'
def verify():
 for p,h in PINS.items():
  if hashlib.sha256((R/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozenbeta source changed: '+p)
def load(n,path):
 s=importlib.util.spec_from_file_location(n,R/path);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
verify();oldpath=list(sys.path);aliases=['beta','kernel','tensor','current','vacuum_cc','free_baryons','lepton_reference','events'];missing=object();old={n:sys.modules.get(n,missing)for n in aliases}
try:
 L=load('hotbeta_spectral_leptons','thermodynamics/lepton_reference.py');sys.modules['lepton_reference']=L
 E=load('hotbeta_spectral_EOS','relativistic_free_eos/free_baryons.py');sys.modules['free_baryons']=E
 V=load('hotbeta_spectral_vacuum','muon_vacuum/vacuum_cc.py');sys.modules['vacuum_cc']=V
 C=load('hotbeta_spectral_current','muon_full_current/current.py');sys.modules['current']=C
 T=load('hotbeta_spectral_tensor','relativistic_tensor/tensor.py');sys.modules['tensor']=T
 K=load('hotbeta_spectral_crossed','relativistic_crossed_thermal/kernel.py');sys.modules['kernel']=K
 EV=load('hotbeta_spectral_eventspinors','relativistic_capture_oracle/events.py');sys.modules['events']=EV
 B=load('hotbeta_spectral_restcurrent','relativistic_crossed_contract/beta.py');sys.modules['beta']=B
 M=load('hotbeta_spectral_totalfunctions','hot_beta_moments/hot_moments.py')
finally:
 sys.path[:]=oldpath
 for n,v in old.items():
  if v is missing:sys.modules.pop(n,None)
  else:sys.modules[n]=v
MN=E.MN;MP=E.MP;ME=L.ELECTRON_MASS;HC=E.HBARC;HBAR=B.HBAR
ENDPOINT=(MN-MP-ME)*(MN+MP+ME)/(2*MN)
FIXED=dict(T=2.,mu_n=919.8287578100662,mu_p=916.8449303084382,mu_e=.17446333200826397)
def real(x):return isinstance(x,numbers.Real)and not isinstance(x,(bool,np.bool_))and math.isfinite(x)
def represented(log):
 if log is None:return dict(value=0.,log=None,representable=True,reason='empty_support')
 if not real(log):raise ArithmeticError('Finitephysical log required')
 try:v=math.exp(log)
 except OverflowError:return dict(value=None,log=float(log),representable=False,reason='overflow')
 return dict(value=v if v else None,log=float(log),representable=v>0,reason=None if v else'underflow')
def matched():
 verify();b,meta=K.matched_npe(2.,1e-7,.3)
 if vars(b)!=FIXED:raise RuntimeError('Fixedphysicalstate densityroots changed')
 return b,meta

def integrate(Enu,*,parent_order=10,cone_order=12,electron_order=12,phi_order=8,tail=60.,budget_seconds=105.):
 verify()
 if not real(Enu)or not 0<Enu<=1.4 or not real(tail)or not 30<=tail<=100 or not real(budget_seconds)or not 0<budget_seconds<=120:raise ValueError('Finiteincident0<E<=1.4,tail30..100,budget0..120 required; notuniformdomaincertificate')
 if any(type(n)is not int or n<4 for n in[parent_order,cone_order,electron_order,phi_order]):raise ValueError('Literalintegerorders>=4 required')
 b,meta=matched();start=time.monotonic();x,w=leggauss(parent_order);ct,cw=leggauss(cone_order);et,ew=leggauss(electron_order)
 # Exact minimalparentrapidity only for labenergiesaboverestendpoint.
 eta0=max(0.,math.log(Enu/ENDPOINT));pmin=MN*math.sinh(eta0);kmin=2*MN*math.sinh(eta0/2)**2
 upper=max(kmin,0.,b.mu_n-MN,b.mu_p+b.mu_e+Enu-MN)+tail*b.T;pmax=math.sqrt(upper*(2*MN+upper));scale=math.sqrt(2*MN*b.T)
 cuts=sorted({pmin,pmax}|{v*scale for v in[.4,1.,2.,3.5,5.,7.]if pmin<v*scale<pmax})
 if Enu<ENDPOINT:
  switch=MN*math.sinh(math.log(ENDPOINT/Enu))
  if pmin<switch<pmax:cuts=sorted(set(cuts+[switch]))
 logs={'j':[],'chi':[]};density=properdensity=0.;maxfour=maxshell=0.;calls=0;minamp=math.inf
 theta=(ct+1)*math.pi/4;dtheta=cw*math.pi/4;ph=2*math.pi*np.arange(phi_order)/phi_order
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  for p,wp in zip((lo+hi)/2+(hi-lo)*x/2,w*(hi-lo)/2):
   if time.monotonic()-start>budget_seconds:raise TimeoutError('Conditionalactualevent spectrum budget')
   En=math.hypot(MN,p);gamma=En/MN;velocity=p/En;parent=p*p*wp/(math.pi**2*HC**3);density+=parent*float(math.exp(log_expit((b.mu_n-En)/b.T)));properdensity+=parent/gamma*float(math.exp(log_expit((b.mu_n-En)/b.T)))
   clo=max(-1.,(Enu/(gamma*ENDPOINT)-1)/velocity)
   if clo>=1:continue
   cs=clo+(1-clo)*np.sin(theta)**2;wc=(1-clo)*2*np.sin(theta)*np.cos(theta)*dtheta/2
   # Labdelta integrates dE* and contributes1/[gamma(1+vcos*)].
   denom=gamma*(1+velocity*cs);restnu=Enu/denom;dS=2*MN*(ENDPOINT-restnu);S=MN*MN-2*MN*restnu
   ps=np.sqrt(dS*(dS+4*MP*ME)/(4*S));rt=np.sqrt(S);Es=np.hypot(ps,ME);gpair=(MN-restnu)/rt;vpair=restnu/(MN-restnu)
   EE=gpair[:,None]*(Es[:,None]-vpair[:,None]*ps[:,None]*et[None,:]);px=ps[:,None]*np.sqrt(1-et[None,:]**2);pz=gpair[:,None]*(ps[:,None]*et[None,:]-vpair[:,None]*Es[:,None]);N=cone_order*electron_order
   pn=np.tile([MN,0.,0.,0.],(N,1));nu=np.column_stack([np.repeat(restnu,electron_order),np.zeros(N),np.zeros(N),np.repeat(restnu,electron_order)]);electron=np.column_stack([EE.ravel(),px.ravel(),np.zeros(N),pz.ravel()]);proton=pn-electron-nu
   amp=B.amplitude(pn,proton,electron,nu);minamp=min(minamp,float(np.min(amp)))
   if np.any(amp<=0)or not np.all(np.isfinite(amp)):raise ArithmeticError('Positive fullbeta amplitude unresolved')
   # dEe/dzeta=Eanti* ps/sqrtS; exactspinavgrestDalitzphase.
   phase=(wc[:,None]/denom[:,None]*(restnu*ps/rt)[:,None]*ew[None,:]).ravel()*amp/(64*math.pi**3*MN*HBAR)*parent/gamma
   if np.any(phase<=0):raise ArithmeticError('Positiveeventphase unresolved')
   cc=np.repeat(cs,electron_order);si=np.sqrt(1-cc*cc)
   def labenergy(v4):
    zz=-si[:,None]*v4[:,1,None]*np.cos(ph)[None,:]+cc[:,None]*v4[:,3,None]
    return gamma*(v4[:,0,None]+velocity*zz)
   Ep=labenergy(proton);El=labenergy(electron);Ev=labenergy(nu);maxfour=max(maxfour,float(np.max(abs(En-Ep-El-Ev))),float(np.max(abs(Ev-Enu))));maxshell=max(maxshell,float(np.max(abs(proton[:,0]**2-np.sum(proton[:,1:]**2,axis=1)-MP*MP)))/MP**2)
   fn=float(log_expit((b.mu_n-En)/b.T));nfn=float(log_expit((En-b.mu_n)/b.T));fp=log_expit((b.mu_p-Ep)/b.T);nfp=log_expit((Ep-b.mu_p)/b.T);fe=log_expit((b.mu_e-El)/b.T);nfe=log_expit((El-b.mu_e)/b.T)
   base=np.log(phase[:,None]/phi_order);logs['j'].append(float(logsumexp(base+fn+nfp+nfe)));logs['chi'].append(float(logsumexp(base+nfn+fp+fe)));calls+=N*phi_order
 dos=Enu*Enu/(2*math.pi**2*HC**3);out={k:represented(float(logsumexp(v))-math.log(dos)if v else None)for k,v in logs.items()};mu=b.mu_n-b.mu_p-b.mu_e
 return dict(**out,relaxation=out['j']['value']+out['chi']['value']if out['j']['representable']and out['chi']['representable']else None,Enu=Enu,bath=vars(b),matched_state=meta,rest_endpoint=ENDPOINT,parent_minimum_momentum=pmin,parent_minimum_kinetic=kmin,parent_maximum_kinetic=upper,parent_density_on_allowed_domain=density,parent_proper_density_on_allowed_domain=properdensity,parent_panels=cuts,one_helicity_DOS=dos,max_conditional_energy_residual=maxfour,max_proton_mass_relative_residual=maxshell,min_amplitude=minamp,log_kms_expected=-(Enu-mu)/b.T,log_kms_actual=out['j']['log']-out['chi']['log']if out['j']['log']is not None and out['chi']['log']is not None else None,equilibrium_mu_antinue=mu,controls=dict(parent_order=parent_order,cone_order=cone_order,electron_order=electron_order,phi_order=phi_order,tail=tail),events=calls,seconds=time.monotonic()-start,source_pins=PINS,implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),units='j and truechi s^-1 peroneactive antinueoccupation; DOS E²/(2pi²HC³)',channel='n <-> p+e-+antinue ONLY; noPCjoin')
