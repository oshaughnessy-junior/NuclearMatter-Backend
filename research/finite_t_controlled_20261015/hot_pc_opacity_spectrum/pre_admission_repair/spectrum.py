"""Fixed matched physical free-npe positron capture/inverse spectral j and TRUE chi.

Phase-coordinate code derived from frozen hot_ec_opacity_spectrum/spectrum.py;
crossed amplitude/occupations/normalization independently gated before integration.

Direct independent FD occupation products in exact particle-hole phase space.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util,sys,math,hashlib,time,json
from dataclasses import asdict
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit,logsumexp
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
SOURCE_PATHS=['hot_electron_capture_moments/electron_capture.py','relativistic_capture_oracle/events.py','relativistic_free_eos/free_baryons.py','thermodynamics/lepton_reference.py','relativistic_tensor/tensor.py','muon_full_current/current.py','muon_vacuum/vacuum_cc.py']
PINS={'hot_electron_capture_moments/electron_capture.py': 'b9954da0a2324376c19e4277013436b135d1432f9cb9904b03761be30bd1b3d1', 'relativistic_capture_oracle/events.py': 'c2c4afc2797b812c0585cdf09c598553e45c275ff362b64277b31350d49c4a76', 'relativistic_free_eos/free_baryons.py': 'aa5fc335de62dde0aa33e328410560741a55ce9dcd5fcbf8b3f4a625e5ffed5b', 'thermodynamics/lepton_reference.py': 'f87ea4bcf8830174995bf9c7061a943052c2d3be8d4386befe3c90215ef97ec9', 'relativistic_tensor/tensor.py': '6fd5feeb66eae2c2685b0803eb27b2496a6f035f570c83d4f4689429e307d975', 'muon_full_current/current.py': '447191554f7e0d239c57f17ab3208f0632dcc51c72b617738785d73d328b2936', 'muon_vacuum/vacuum_cc.py': '7f79ba2f60f90b53cd23c7cc19e9191837b99442a5fb95f298d163d20d372dd9'}
PINS.update({'review_pc_moments/reference.py':'2ca34b181865df529ee0b0d599faa932760190cfc25f97d71d513db6290ef133','positron_capture/pc_kernel.py':'6291c802fe8e4b86572c2c7582df28dd9518980500ef67b2d7dec99dc3eeb4e6'})
def verify():
 for p,h in PINS.items():
  if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen physical source changed: '+p)
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
verify();oldpath=list(sys.path);aliases=['lepton_reference','current','vacuum_cc'];missing=object();old={a:sys.modules.get(a,missing)for a in aliases}
try:
 L=load('hot_PC_spectral_leptons','thermodynamics/lepton_reference.py');sys.modules['lepton_reference']=L
 C=load('hot_PC_spectral_total_current','hot_electron_capture_moments/electron_capture.py')
 V=load('hot_PC_spectral_vacuum','muon_vacuum/vacuum_cc.py');sys.modules['vacuum_cc']=V
 J=load('hot_PC_spectral_current','muon_full_current/current.py');sys.modules['current']=J
 K=load('hot_PC_spectral_tensor','relativistic_tensor/tensor.py')
finally:
 sys.path[:]=oldpath
 for a,v in old.items():
  if v is missing:sys.modules.pop(a,None)
  else:sys.modules[a]=v
P=load('hot_PC_spectral_scalar_reference','review_pc_moments/reference.py')
HC=C.HC;HBAR=C.HBAR
FIXED_BATH=dict(T=2.,mu_n=919.8287578100662,mu_p=916.8449303084382,mu_e=.17446333200826397)

def real(x):return type(x)in(int,float)and math.isfinite(x)
def represented(log):
 if log is None:return dict(value=0.,log=None,representable=True,reason='empty_support')
 if not math.isfinite(log):raise ArithmeticError('Finite physical log required')
 try:v=math.exp(log)
 except OverflowError:return dict(value=None,log=log,representable=False,reason='overflow')
 return dict(value=v if v else None,log=log,representable=v>0,reason=None if v else'underflow')
def matched():
 verify();b,meta=C.matched(2.,1e-7,.3)
 if asdict(b)!=FIXED_BATH:raise RuntimeError('Fresh EOS root no longer matches fixed physical state')
 return b,meta
def differential(bath,Enu,El,c,*,energy_order=12,phi_order=4,tail=50.):
 verify()
 if not isinstance(bath,C.Bath)or asdict(bath)!=FIXED_BATH or not all(real(v)for v in[Enu,El,c,tail])or Enu<=0 or El<=C.ME or not -1<=c<=1 or not 30<=tail<=100:raise ValueError('Physical finite differential point and tail30..100 required')
 if type(energy_order)is not int or energy_order<8 or type(phi_order)is not int or phi_order<4:raise ValueError('Integer energy>=8,phi>=4 required')
 pe=math.sqrt((El-C.ME)*(El+C.ME));w=Enu-El;q=math.sqrt((Enu-pe)**2+2*Enu*pe*(1-c));intervals=K.support(C.MP,C.MN,w,q);xs,ws=leggauss(energy_order);ea=[];ew=[];panels=[]
 for seg in intervals:
  lo=seg['lower'];hi=max(lo,C.MP,bath.mu_p,bath.mu_n-w)+tail*bath.T
  if seg['upper']is not None:hi=min(hi,seg['upper'])
  if hi<=lo:continue
  cuts=sorted({lo,hi}|{v for center in[C.MP,bath.mu_p,bath.mu_n-w]for h in[0,2,6,12,24]if lo<(v:=center+h*bath.T)<hi})
  for a,b in zip(cuts[:-1],cuts[1:]):
   # Distinct real edges can collapse in floating transforms; no log(0) panel.
   if not b>a:continue
   ea.extend(a+(xs+1)*(b-a)/2);ew.extend(ws*(b-a)/2);panels.append([a,b])
 if not ea:return dict(j=represented(None),chi=represented(None),support=intervals,panels=panels,points=0)
 ea=np.repeat(np.array(ea),phi_order);wt=np.repeat(np.array(ew),phi_order)/phi_order;ph=np.tile(2*math.pi*np.arange(phi_order)/phi_order,len(ea)//phi_order);pn=np.sqrt((ea-C.MP)*(ea+C.MP));cos=((C.MP-C.MN)*(C.MP+C.MN)+w*w-q*q+2*w*ea)/(2*pn*q)
 if np.any(abs(cos)>1+1e-10):raise ArithmeticError('Exact support angle outside domain')
 cos=np.clip(cos,-1,1);pt=pn*np.sqrt(1-cos*cos);n=np.column_stack([ea,pt*np.cos(ph),pt*np.sin(ph),pn*cos]);p=n+np.array([w,0,0,q]);kz=(Enu*Enu+q*q-pe*pe)/(2*q);kx2=Enu*Enu-kz*kz
 if kx2< -1e-10*Enu*Enu:raise ArithmeticError('Lepton shell outside domain')
 kx=math.sqrt(max(0.,kx2));k=np.tile([Enu,kx,0,kz],(len(ea),1));l=np.tile([El,kx,0,kz-q],(len(ea),1));amp=P.amplitude(k,n,l,p)
 if np.any(amp<=0)or not np.all(np.isfinite(amp)):raise ArithmeticError('Positive physical current contraction unresolved')
 # scalar amplitude averages FOUR n,e+ spins for PC emission.
 # Reciprocal unaveraged spin sum is shared by absorption, with ONE anti helicity. This differential coefficient
 # prefactor is obtained from the unaveraged W convention, not fitted to totals.
 base=np.log(wt*amp*pe/(32*math.pi**3*q*Enu*HBAR));lf=log_expit((bath.mu_n-ea-w)/bath.T)+log_expit((ea-bath.mu_p)/bath.T)+log_expit((-bath.mu_e-El)/bath.T);lb=log_expit((bath.mu_p-ea)/bath.T)+log_expit((ea+w-bath.mu_n)/bath.T)+log_expit((El+bath.mu_e)/bath.T)
 return dict(j=represented(float(logsumexp(base+lf))),chi=represented(float(logsumexp(base+lb))),support=intervals,panels=panels,points=len(ea))

def integrate(Enu,*,lepton_order=8,angle_order=12,energy_order=12,phi_order=4,tail=50.,outer_tail=30.,budget_seconds=110.):
 verify()
 if not real(Enu)or not 0<Enu<=60 or not all(real(v)for v in[outer_tail,budget_seconds])or not 20<=outer_tail<=60 or not 0<budget_seconds<=120:raise ValueError('Finite E0..60,outertail20..60,budget0..120 required')
 if type(lepton_order)is not int or lepton_order<4 or type(angle_order)is not int or angle_order<4:raise ValueError('Integer lepton/angle orders>=4')
 if type(energy_order)is not int or energy_order<8 or type(phi_order)is not int or phi_order<4 or not real(tail)or not 30<=tail<=100:raise ValueError('Energy>=8,phi>=4 and finite tail30..100 required')
 b,meta=matched();mid=Enu+C.MP-C.MN;p0=math.sqrt(max(0.,mid*mid-C.ME*C.ME));width=max(.03*b.T,(Enu+p0)*math.sqrt(b.T/C.MP),(Enu+p0)**2/(2*C.MP));upper=max(mid,C.ME,-b.mu_e)+outer_tail*b.T
 cuts=sorted({C.ME,upper}|{v for center in[mid,-b.mu_e]for scale in[b.T,width]for h in[-12,-6,-3,-1,-.5,0,.5,1,3,6,12]if C.ME<(v:=center+h*scale)<upper});x,w=leggauss(lepton_order);cs,cw=leggauss(angle_order);logs={'j':[],'chi':[]};calls=empty=timelike=0;start=time.monotonic()
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  xl=math.sqrt(lo-C.ME);xh=math.sqrt(hi-C.ME)
  if not xh>xl:continue
  for a,ww in zip(x,w):
   y=xl+(a+1)*(xh-xl)/2;El=C.ME+y*y;weight=ww*(xh-xl)*y
   for c,v in zip(cs,cw):
    if time.monotonic()-start>budget_seconds:raise TimeoutError('Actual fixed-state spectral integration budget stop')
    r=differential(b,Enu,float(El),float(c),energy_order=energy_order,phi_order=phi_order,tail=tail);calls+=1
    timelike+=int((Enu-El)**2>((Enu-math.sqrt(El*El-C.ME*C.ME))**2+2*Enu*math.sqrt(El*El-C.ME*C.ME)*(1-c)))
    empty+=r['j']['log']is None
    for key in logs:
     if r[key]['log']is not None:logs[key].append(r[key]['log']+math.log(weight*v))
 out={key:represented(float(logsumexp(v))if v else None)for key,v in logs.items()};mu=b.mu_n-b.mu_p-b.mu_e
 return dict(**out,relaxation=out['j']['value']+out['chi']['value']if out['j']['representable']and out['chi']['representable']else None,Enu=Enu,bath=asdict(b),matched_state=meta,equilibrium_mu_antinue=mu,log_kms_expected=-(Enu-mu)/b.T,log_kms_actual=out['j']['log']-out['chi']['log']if out['j']['log']is not None and out['chi']['log']is not None else None,controls=dict(lepton_order=lepton_order,angle_order=angle_order,energy_order=energy_order,phi_order=phi_order,tail=tail,outer_tail=outer_tail),panels_Eelectron=cuts,upper_Eelectron=upper,calls=calls,empty_nodes=empty,timelike_nodes=timelike,seconds=time.monotonic()-start,source_pins=PINS,units='j and true chi: s^-1 per one active antinue occupation; relaxation=j+chi',channel='n+e+ <-> p+antinue ONLY; no joined inventory',implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
