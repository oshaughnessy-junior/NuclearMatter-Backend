"""Independent beta spectral phase: rapidity, flat Dalitz, orientation delta.

The actual integral uses Dirac traces; no KMS inverse or author phase routine.
"""
from pathlib import Path
import importlib.util,sys,json,math,time,hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit,log_expit,logsumexp
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('review_hot_beta_spectrum_frozen_functions',ROOT/'hot_beta_opacity_spectrum/spectrum.py');S=importlib.util.module_from_spec(s);sys.modules[s.name]=S;s.loader.exec_module(S)
E=S.EV;MN=S.MN;MP=S.MP;ME=S.ME;HC=S.HC;HBAR=S.HBAR;ENDPOINT=(MN*MN-(MP+ME)**2)/(2*MN)
# Stable endpoint is recomputed from factored mass differences.
ENDPOINT=(MN-MP-ME)*(MN+MP+ME)/(2*MN)
def slash(v,m=0):return np.einsum('nu,u,uab->nab',v,E.MET,E.G)+m*E.I
def trace_amp(n,p,l,k):
 V=E.vertices(p-n,(MN+MP)/2);AV=E.adj(V);L=E.G@(E.I-E.G5);AL=E.adj(L[None,...])[0]
 H=np.einsum('nab,nubc,ncd,nvda->nuv',slash(p,MP),V,slash(n,MN),AV,optimize=True)
 LT=np.einsum('nab,ubc,ncd,vda->nuv',slash(l,ME),L,slash(k),AL,optimize=True)
 val=np.einsum('nuv,nuv,u,v->n',LT,H,E.MET,E.MET)
 if np.max(abs(val.imag))>1e-10*np.max(abs(val.real)):raise ArithmeticError('Trace imaginary residual')
 return E.GF**2*E.VUD**2/4*val.real

def integral(Enu,parent=8,nu_order=10,electron=10,phi_order=8,tail=65.,budget=110.):
 start=time.monotonic();b,meta=S.matched();x,w=leggauss(parent);v,vw=leggauss(nu_order);z,zw=leggauss(electron);ph=2*math.pi*np.arange(phi_order)/phi_order
 eta0=max(0.,math.log(Enu/ENDPOINT));upper=math.acosh(1+(MN*(math.cosh(eta0)-1)+tail*b.T)/MN)
 edges=sorted({eta0,upper}|{t for t in [.02,.05,.09,.15,.23,.35,.5,.7]if eta0<t<upper})
 switch=abs(math.log(ENDPOINT/Enu))
 if Enu<ENDPOINT and eta0<switch<upper:edges=sorted(set(edges+[switch]))
 logs=dict(j=[],chi=[]);density=0.;four=0.;points=0
 for a,bb in zip(edges[:-1],edges[1:]):
  for eta,we in zip((a+bb)/2+(bb-a)*x/2,w*(bb-a)/2):
   if time.monotonic()-start>budget:raise TimeoutError('Independent spectral phase bounded110s')
   gamma=math.cosh(eta);sh=math.sinh(eta);En=MN*gamma;velocity=math.tanh(eta);pparent=MN*sh
   density+=MN**3*sh*sh*gamma*we/(math.pi**2*HC**3)*expit((b.mu_n-En)/b.T)
   # Delta over cos(theta_nu,parent) => 1/(2 sinh(eta) E*).
   lo=Enu*math.exp(-eta);hi=min(ENDPOINT,Enu*math.exp(eta))
   if hi<=lo:continue
   # t² distance from upper rest endpoint smooths flat-Dalitz edge support.
   t=(v+1)/2;en=hi-(hi-lo)*t*t;wn=(hi-lo)*t*vw
   A=MN-en;ss=MN*MN-2*MN*en;B=(ss+ME*ME-MP*MP)/2
   dd=2*MN*(ENDPOINT-en);disc=.5*np.sqrt(dd*(dd+4*MP*ME));center=A*B/ss;half=en*disc/ss
   ee=center[:,None]+half[:,None]*z[None,:];welectron=half[:,None]*zw[None,:]
   nu=np.repeat(en,electron);el=ee.ravel();pe=np.sqrt((el-ME)*(el+ME));cos=(np.repeat(B,electron)-np.repeat(A,electron)*el)/(nu*pe)
   if np.max(abs(cos))>1+2e-6:raise ArithmeticError('Independent Dalitz electron cone outside support')
   cos=np.clip(cos,-1,1);N=len(el);n=np.tile([MN,0.,0.,0.],(N,1));k=np.column_stack([nu,nu*0,nu*0,nu]);l=np.column_stack([el,pe*np.sqrt(1-cos*cos),el*0,pe*cos]);p=n-l-k;amp=trace_amp(n,p,l,k)
   if np.min(amp)<=0:raise ArithmeticError('Independent positive trace amplitude')
   measure=MN**3*sh*sh*gamma*we/(math.pi**2*HC**3)/gamma
   weight=(wn[:,None]*welectron).ravel()*amp/(64*math.pi**3*MN*HBAR)*measure/(2*sh*nu)
   cn=(Enu/(gamma*nu)-1)/velocity
   if np.max(abs(cn))>1+1e-10:raise ArithmeticError('Independent lab-energy cone outside support')
   cn=np.clip(cn,-1,1);sn=np.sqrt(1-cn*cn)
   def labenergy(vec):return gamma*(vec[:,0,None]+velocity*(cn[:,None]*vec[:,3,None]-sn[:,None]*vec[:,1,None]*np.cos(ph)[None,:]))
   ep=labenergy(p);elab=labenergy(l);ev=labenergy(k);four=max(four,float(np.max(abs(En-ep-elab-ev))),float(np.max(abs(ev-Enu))))
   # Both occupations are actual separate products. No neutrino factor here.
   fn=float(log_expit((b.mu_n-En)/b.T));nv=float(log_expit((En-b.mu_n)/b.T));fp=log_expit((b.mu_p-ep)/b.T);pv=log_expit((ep-b.mu_p)/b.T);fe=log_expit((b.mu_e-elab)/b.T);lv=log_expit((elab-b.mu_e)/b.T)
   logbase=np.log(weight[:,None]/phi_order);logs['j'].append(float(logsumexp(logbase+fn+pv+lv)));logs['chi'].append(float(logsumexp(logbase+nv+fp+fe)));points+=N*phi_order
 dos=Enu*Enu/(2*math.pi**2*HC**3)
 return dict(Enu=Enu,j_log=float(logsumexp(logs['j']))-math.log(dos),chi_log=float(logsumexp(logs['chi']))-math.log(dos),density_allowed=density,max_energy_residual=four,points=points,controls=dict(parent=parent,nu=nu_order,electron=electron,phi=phi_order,tail=tail),rapidity_edges=edges,seconds=time.monotonic()-start)
