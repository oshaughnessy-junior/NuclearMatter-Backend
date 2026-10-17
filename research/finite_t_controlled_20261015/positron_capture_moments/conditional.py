"""Fixed outgoing Enu via CM boost-cone delta, distinct from q-axis response."""
import math,time
import numpy as np
from scipy.optimize import brentq
from scipy.special import log_expit,logsumexp
from numpy.polynomial.legendre import leggauss
import moments as M

def spectrum(b,Enu,norder=12,corder=12,eorder=16,phi_order=8,tail=40.):
 if not isinstance(b,M.PC.crossed.Bath)or not .2<=b.T<=4 or max((b.mu_n-M.MN)/b.T,(b.mu_p-M.MP)/b.T)>-5 or abs(b.mu_e)/b.T>3:raise ValueError('Selected dilute pair-rich physical npe domain required')
 if not math.isfinite(Enu)or Enu<=0:raise ValueError('Positive finite selected Enu required')
 for x in [norder,corder,eorder,phi_order]:
  if isinstance(x,bool)or not isinstance(x,int)or x<4:raise ValueError('Integer orders>=4')
 if not math.isfinite(tail)or not 20<=tail<=100:raise ValueError('Finite tail20..100')
 kn,wn=M.grid(norder,b.T,tail*b.T);cs,ws=leggauss(corder);xx,ww=leggauss(eorder);ph=2*math.pi*np.arange(phi_order)/phi_order;logs=[[],[]];start=time.monotonic();events=0
 eb=M.E.Bath(b.T,M.MP,M.MN,b.mu_p,b.mu_n,-b.mu_e,anti=True,ml=M.ME,Mweak=M.MW)
 for kin,wni in zip(kn,wn):
  en=M.MN+kin;pn=math.sqrt(kin*(kin+2*M.MN));fn=log_expit((b.mu_n-en)/b.T);nfn=log_expit((en-b.mu_n)/b.T)
  for ci,wci in zip(cs,ws):
   def ends(ke):
    ee=M.ME+ke;pe=math.sqrt(ke*(ke+2*M.ME));ptot=math.sqrt(pn*pn+pe*pe+2*pn*pe*ci);etot=en+ee;s=etot*etot-ptot*ptot;pf=(s-M.MP*M.MP)/(2*math.sqrt(s));return pf*(etot-ptot)/math.sqrt(s),pf*(etot+ptot)/math.sqrt(s)
   upper=tail*b.T
   if ends(upper)[1]<Enu or ends(0)[0]>Enu:continue
   lo=0. if ends(0)[1]>=Enu else brentq(lambda z:ends(z)[1]-Enu,0,upper,xtol=2e-13)
   hi=upper if ends(upper)[0]<=Enu else brentq(lambda z:ends(z)[0]-Enu,lo,upper,xtol=2e-13)
   ee=M.ME+(lo+hi)/2+(hi-lo)*xx/2;ew=ww*(hi-lo)/2
   EE,PH=np.meshgrid(ee,ph,indexing='ij');ee=EE.ravel();phv=PH.ravel();ew=np.repeat(ew,phi_order)*2*math.pi/phi_order;pe=np.sqrt((ee-M.ME)*(ee+M.ME));N=len(ee)
   nv=np.tile([en,0,0,pn],(N,1));lv=np.column_stack([ee,pe*math.sqrt(1-ci*ci),np.zeros(N),pe*ci]);P=nv+lv;s=P[:,0]**2-np.sum(P[:,1:]**2,axis=1);rt=np.sqrt(s);beta=P[:,1:]/P[:,0,None];speed=np.linalg.norm(beta,axis=1);gamma=P[:,0]/rt;axis=beta/speed[:,None];e2=np.tile([0,1,0],(N,1));e1=np.cross(e2,axis);pf=(s-M.MP*M.MP)/(2*rt);co=(Enu/(gamma*pf)-1)/speed
   if np.max(abs(co))>1+1e-10:raise ArithmeticError('Boost-cone energy support lost')
   co=np.clip(co,-1,1);direc=co[:,None]*axis+np.sqrt(1-co*co)[:,None]*(np.cos(phv)[:,None]*e1+np.sin(phv)[:,None]*e2);k=M.E.boost(np.column_stack([pf,pf[:,None]*direc]),beta);pv=M.E.boost(np.column_stack([(s+M.MP*M.MP)/(2*rt),-pf[:,None]*direc]),beta)
   pi=np.sqrt((s-(M.MN+M.ME)**2)*(s-(M.MN-M.ME)**2))/(2*rt);amp=M.E.amplitude(eb,k,pv,lv,nv)/2;sigma=amp/(64*math.pi**2*s)*pf/pi;flux=pi*rt/(en*ee);weight=pn*en*wni/math.pi**2*pe*ee/(2*math.pi**2)*wci*ew*flux*sigma/(M.HBAR*M.HC**3)/(gamma*speed*pf)
   fp=log_expit((b.mu_p-pv[:,0])/b.T);nfp=log_expit((pv[:,0]-b.mu_p)/b.T);fl=log_expit((-b.mu_e-ee)/b.T);nfl=log_expit((ee+b.mu_e)/b.T)
   logs[0].append(float(logsumexp(np.log(weight)+fn+fl+nfp)));logs[1].append(float(logsumexp(np.log(weight)+nfn+nfl+fp)));events+=N
 dos=Enu**2/(2*math.pi**2*M.HC**3)
 return dict(Enu=Enu,j=M.PC.crossed.represented(float(logsumexp(logs[0]))-math.log(dos)),chi=M.PC.crossed.represented(float(logsumexp(logs[1]))-math.log(dos)),DOS=dos,seconds=time.monotonic()-start,events=events,controls=dict(norder=norder,corder=corder,eorder=eorder,phi_order=phi_order,tail=tail),measure='dR/dEnu divided by one-helicity DOS; CMcos delta Jacobian 1/(gamma beta Enu_star)')
