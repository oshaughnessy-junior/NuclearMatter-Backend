"""Independent Q-rest event and vacuum Dalitz references; shared current primitives."""
from pathlib import Path
import sys,math
import numpy as np
from scipy.special import log_expit,logsumexp
sys.dont_write_bytecode=True
from kernel import *
sys.path.insert(0,str(ROOT/'relativistic_crossed_contract'));import beta as b
def event_reference(bath,Enu,Ee,c,order=80):
 """Positive-timelike Q-rest sphere; explicit crossed external spinors.

 Independent support/Jacobian: dEa/q = pstar/sqrt(s) dcosstar.
 """
 pe=math.sqrt((Ee-ME)*(Ee+ME));q=math.sqrt((Enu-pe)**2+2*Enu*pe*(1+c));om=Enu+Ee;s=ME*ME+2*Enu*(Ee-pe*c);rt=math.sqrt(s)
 if s>(MN-MP)**2:return [None,None]
 Estar=((MN-MP)*(MN+MP)-s)/(2*rt);pstar=math.sqrt((Estar-MP)*(Estar+MP));gamma=om/rt;vel=q/om
 x,w=leggauss(order)
 # The Q-rest sphere is highly boosted: thermal baryons occupy a thin polar
 # cap. Resolve in independently derived affine Ep(cstar), not uniform cos.
 cuts=[-1.,1.]
 for Ecut in [MP+k*bath.T for k in [0,1,3,8,20,40,70]]:
  cut=(Ecut/gamma-Estar)/(vel*pstar)
  if -1<cut<1:cuts.append(cut)
 cuts=sorted(cuts);xx=[];ww=[]
 for a,z in zip(cuts[:-1],cuts[1:]):
  xx.extend(a+(x+1)*(z-a)/2);ww.extend(w*(z-a)/2)
 x=np.array(xx);w=np.array(ww)
 ph=2*math.pi*np.arange(8)/8;X,P=np.meshgrid(x,ph,indexing='ij');X=X.ravel();P=P.ravel();weights=np.repeat(w/8,8)
 pa=np.column_stack([np.full(len(X),Estar),pstar*np.sqrt(1-X*X)*np.cos(P),pstar*np.sqrt(1-X*X)*np.sin(P),pstar*X]);pb=pa.copy();pb[:,0]+=rt
 boost=np.tile([0.,0.,vel],(len(X),1));pa=b.boost(pa,boost);pb=b.boost(pb,boost)
 kz=(q*q+Enu*Enu-pe*pe)/(2*q);kx=math.sqrt(max(0.,Enu*Enu-kz*kz));k=np.tile([Enu,kx,0.,kz],(len(X),1));l=np.tile([Ee,-kx,0.,q-kz],(len(X),1))
 amp=b.amplitude(pb,pa,l,k,inverse=True)
 pref=pe*pstar/(64*math.pi**3*Enu*rt*t.HBAR)
 logp=log_expit((bath.mu_p-pa[:,0])/bath.T);vacp=log_expit((pa[:,0]-bath.mu_p)/bath.T);logn=log_expit((bath.mu_n-pb[:,0])/bath.T);vacn=log_expit((pb[:,0]-bath.mu_n)/bath.T)
 lc=math.log(pref)+logsumexp(np.log(weights)+np.log(amp)+logp+vacn)+log_expit((bath.mu_e-Ee)/bath.T)
 lj=math.log(pref)+logsumexp(np.log(weights)+np.log(amp)+logn+vacp)+log_expit((Ee-bath.mu_e)/bath.T)
 shell=max(np.max(abs(pa[:,0]**2-np.sum(pa[:,1:]**2,axis=1)-MP*MP)),np.max(abs(pb[:,0]**2-np.sum(pb[:,1:]**2,axis=1)-MN*MN)))/(MN*MN)
 return [float(lj),float(lc),float(shell)]

def rest_spectrum(Enu,order=32,model='full'):
 """Exact independent neutron-rest dGamma/dEnu, Dalitz energy coordinate."""
 S=MN*MN-2*MN*Enu;rt=math.sqrt(S);Es=(S+ME*ME-MP*MP)/(2*rt);ps=math.sqrt((Es-ME)*(Es+ME));gm=(MN-Enu)/rt;vel=Enu/(MN-Enu);lo,hi=gm*(Es-vel*ps),gm*(Es+vel*ps)
 x,w=leggauss(order);E=lo+(x+1)*(hi-lo)/2;p=np.sqrt((E-ME)*(E+ME));c=((MN-E)*Enu-MN*((MN*MN-MP*MP+ME*ME)/(2*MN)-E))/(-p*Enu)
 if np.max(abs(c))>1+1e-8:raise ArithmeticError('Rest spectrum shell outside support')
 pn=np.tile([MN,0.,0.,0.],(order,1));k=np.tile([Enu,0.,0.,Enu],(order,1));l=np.column_stack([E,p*np.sqrt(np.maximum(0.,1-c*c)),np.zeros(order),p*c]);pp=pn-l-k;A=b.amplitude(pn,pp,l,k,model=model)
 return float(np.sum(w*A)*(hi-lo)/2/(64*math.pi**3*MN*t.HBAR))

