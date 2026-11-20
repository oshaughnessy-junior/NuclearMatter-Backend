"""Conditional parent energy and exact electron-proton cluster at fixed antinu E."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import math,time,json,hashlib
import numpy as np
from scipy.special import roots_genlaguerre,logsumexp
from numpy.polynomial.legendre import leggauss
from beta_pr55_reference import _weak_current as C
from beta_pr55_reference._weak_quadrature import log_fd
BATH=(2.,919.8287578100662,916.8449303084382,.17446333200826397)
ENDPOINT=(C.MN-C.MP-C.ME)*(C.MN+C.MP+C.ME)/(2*C.MN)
@dataclass(frozen=True)
class Rule:
 radial:int
 cone:int
 electron:int
 phi:int
 family:str
 tail:float=70.
 budget:float=85.
 def __post_init__(self):
  for k in ('radial','cone','electron','phi'):
   if type(getattr(self,k)) is not int or not 4<=getattr(self,k)<=32:raise ValueError('Literal orders4..32')
  if self.family not in ('panels','laguerre'):raise ValueError('Named radial family')
  if isinstance(self.tail,bool) or not math.isfinite(self.tail) or not 50<=self.tail<=100:raise ValueError('Explicit tail50..100')
  if isinstance(self.budget,bool) or not math.isfinite(self.budget) or not 0<self.budget<=85:raise ValueError('Budget<=85s')

def evaluate(E,rule):
 if isinstance(E,bool) or type(E) not in (float,int) or not math.isfinite(E) or not ENDPOINT<E<=1.4:raise ValueError('Selected upper beta E>rest endpoint and<=1.4')
 if type(rule) is not Rule:raise ValueError('Explicit Rule')
 rule.__post_init__();start=time.monotonic();T,mun,mup,mue=BATH
 P=Path(__file__).resolve().parent
 for old,h in json.loads((P/'ORIGINAL_SOURCE_PIN.json').read_text()).items():
  if hashlib.sha256((P/'beta_pr55_reference'/Path(old).name).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source-copy drift')
 r=E/ENDPOINT;eta0=math.log(r);p0=C.MN*math.sinh(eta0);en0=C.MN*math.cosh(eta0)
 xc,wc=leggauss(rule.cone);v=(xc+1)/2;u=1-v*v;cone_weight0=wc*v/2
 xe,we=leggauss(rule.electron);phi=2*math.pi*np.arange(rule.phi)/rule.phi
 if rule.family=='laguerre':
  z,w=roots_genlaguerre(rule.radial,1.5);radial=[(float(a),math.log(float(b)*T)+float(a)-1.5*math.log(float(a))) for a,b in zip(z,w)]
 else:
  x,w=leggauss(rule.radial);radial=[]
  edges=sorted({0.,rule.tail}|{z for z in (1.,2.,4.,8.,16.,32.,50.,70.,90.) if z<rule.tail})
  for a,b in zip(edges[:-1],edges[1:]):
   ya,yb=math.sqrt(a),math.sqrt(b);ys=ya+(x+1)*(yb-ya)/2
   radial.extend((float(y*y),math.log(float(ww)*(yb-ya)*float(y)*T)) for y,ww in zip(ys,w))
 jf=[];cb=[];calls=0;ms=es=dr=recip=0.
 for z,lw in radial:
  if time.monotonic()-start>rule.budget:raise TimeoutError('Bounded parent integration incomplete')
  de=T*z;en=en0+de;p=math.sqrt((en-C.MN)*(en+C.MN));gamma=en/C.MN;vel=p/en
  R=de*(1+(en+en0)/(p+p0));width=R/p
  if not 0<width<2:raise ArithmeticError('Conditional cone not physical')
  cs=1-width*u;D=(en+p-R*u)/C.MN;ev=E/D
  ds=2*C.MN*ENDPOINT*R*(1-u)/(en+p-R*u)
  sep=(C.MP+C.ME)**2+ds;rt=np.sqrt(sep);pr=np.sqrt(ds*(ds+4*C.MP*C.ME))/(2*rt)
  elstar=np.hypot(pr,C.ME);gp=(C.MN-ev)/rt;vp=ev/(C.MN-ev)
  cc,et=np.meshgrid(cs,xe,indexing='ij');ew=np.broadcast_to(we,cc.shape)
  evv=np.broadcast_to(ev[:,None],cc.shape);ppair=np.broadcast_to(pr[:,None],cc.shape)
  ee=gp[:,None]*(elstar[:,None]-vp[:,None]*ppair*et)
  px=ppair*np.sqrt(1-et*et);pz=gp[:,None]*(ppair*et-vp[:,None]*elstar[:,None])
  n=np.tile([C.MN,0,0,0],(cc.size,1));nu=np.column_stack([evv.ravel(),np.zeros(cc.size),np.zeros(cc.size),evv.ravel()])
  lepton=np.column_stack([ee.ravel(),px.ravel(),np.zeros(cc.size),pz.ravel()]);proton=n-lepton-nu
  amp=C.decay_amplitude(n,proton,lepton,nu).reshape(cc.shape)
  if not calls:
   inverse=C.decay_amplitude(n,proton,lepton,nu,inverse=True).reshape(cc.shape)
   recip=float(np.max(abs(inverse/amp-1)))
   if recip>1e-10:raise ArithmeticError('Literal reciprocal current gate')
  mass=float(np.max(abs(C.dot(proton,proton)/C.MP**2-1)));ms=max(ms,mass)
  if mass>1e-10 or np.any(proton[:,0]<C.MP) or np.any(lepton[:,0]<C.ME):raise ArithmeticError('Original future shell guard')
  # Original phase after parent p²dp→p En dEn, time dilation, DOS removal.
  cone_w=width*cone_weight0
  measure=(cone_w[:,None]/D[:,None]*(ev*pr/rt)[:,None]*ew)*amp*p/(32*math.pi**3*E*E*C.HBAR)
  if np.any(measure<=0) or not np.all(np.isfinite(measure)):raise ArithmeticError('Positive phase unresolved')
  base=lw+np.log(measure).ravel()-math.log(rule.phi)
  cflat=cc.ravel();sn=np.sqrt((1-cflat)*(1+cflat))
  zzl=-sn[:,None]*lepton[:,1,None]*np.cos(phi)+cflat[:,None]*lepton[:,3,None]
  zzp=-sn[:,None]*proton[:,1,None]*np.cos(phi)+cflat[:,None]*proton[:,3,None]
  el=gamma*(lepton[:,0,None]+vel*zzl);ep=gamma*(proton[:,0,None]+vel*zzp)
  conserved=float(np.max(abs(en-ep-el-E)))/en;es=max(es,conserved)
  # Unsquared rest-shell condition before parent boost, independently evaluated.
  shellE=np.sqrt(C.MP*C.MP+np.sum((lepton[:,1:]+nu[:,1:])**2,axis=1))
  delta=float(np.max(abs(C.MN-lepton[:,0]-nu[:,0]-shellE)))/C.MN;dr=max(dr,delta)
  if conserved>1e-10 or delta>1e-10 or np.any(ep<C.MP) or np.any(el<C.ME):raise ArithmeticError('Unsquared delta/energy/future guard')
  fn=float(log_fd((mun-en)/T));hn=float(log_fd((en-mun)/T))
  fp=log_fd((mup-ep)/T);hp=log_fd((ep-mup)/T);fe=log_fd((mue-el)/T);he=log_fd((el-mue)/T)
  jf.append(float(logsumexp(base[:,None]+fn+hp+he)));cb.append(float(logsumexp(base[:,None]+hn+fp+fe)))
  calls+=cc.size*rule.phi
 lj,lc=float(logsumexp(jf)),float(logsumexp(cb))
 return {'E_MeV':E,'bath':dict(zip(('T','mu_n','mu_p','mu_e'),BATH)),'rule':asdict(rule),'parent_min_energy_MeV':en0,'parent_min_momentum_MeV':p0,'log_j_s-1':lj,'log_true_chi_s-1':lc,'j_s-1':math.exp(lj),'true_chi_s-1':math.exp(lc),'max_shell_relative':ms,'max_lab_energy_scaled':es,'max_unsquared_rest_delta_scaled':dr,'literal_current_reciprocity':recip,'KMS_actual_products':lj-lc+(E-(mun-mup-mue))/T,'events':calls,'seconds':time.monotonic()-start,'radial_domain':'semi-infinite' if rule.family=='laguerre' else 'finite parent excess z0..tail'}
