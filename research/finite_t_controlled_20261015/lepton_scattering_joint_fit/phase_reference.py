"""Independent analytic-phi/transverse-shell integrated reference, nu_e BOTH e±."""
import math,time
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
T=2.;MU=.17446333200826397;M=.51099895069;GF=1.1663785e-11;HBAR=6.582119569e-22

def differential(E,F,c,pos,tail=80.):
 om=E-F;Q=math.sqrt(om*om+2*E*F*(1-c));a=2*E*F*(1-c);lo=(-om+Q*math.sqrt(1+4*M*M/a))/2;mu=-MU if pos else MU;hi=max(lo,mu,mu-om)+tail*T;kz=(E*E+Q*Q-F*F)/(2*Q);kx2=max(0.,E*E-kz*kz);h=a/2;v=.96258;g=.5;L,R=(v+g)**2,(v-g)**2
 if pos:L,R=R,L
 pz=lambda e:(2*e*om-a)/(2*Q)
 upper=math.sqrt(max(0.,hi*hi-M*M-pz(hi)**2))
 def integrand(r):
  root=math.sqrt(1+4*(M*M+r*r)/a);e=(-om+Q*root)/2;de=2*Q*r/(a*root);u=E*e-kz*pz(e);z=kx2*r*r/2;amp=L*(u*u+z)+R*((u-h)**2+z)-M*M*(v*v-g*g)*h
  return de*amp*expit((mu-e)/T)*expit((e+om-mu)/T)
 value=quad(integrand,0,upper,epsabs=2e-9,epsrel=3e-10,limit=120)[0]
 return F/(64*math.pi**3*E*Q*HBAR)*16*GF*GF*value

def integrated(E,order=16,angle=24,upper=150.,tail=90.):
 start=time.monotonic();x,w=leggauss(order);z,v=leggauss(angle);total=np.zeros(4)
 for zz,vv in zip((z+1)/2,v/2):
  c=1-2*zz*zz;width=2*E*zz;edges=sorted(set([0.,upper,1.,3.,E,12.,22.,40.,70.]+[E+h*width for h in[-4,-2,-1,-.5,-.2,.2,.5,1,2,4]if 0<E+h*width<upper]));edges=[a for a in edges if a<=upper]
  for lo,hi in zip(edges[:-1],edges[1:]):
   for f,wf in zip((lo+hi)/2+x*(hi-lo)/2,w*(hi-lo)/2):
    if time.monotonic()-start>110:raise TimeoutError('Independent phase anchor bounded work')
    d=sum(differential(E,float(f),float(c),pos,tail)for pos in[False,True]);a=d*wf*vv*4*zz;total+=a*np.array([1,1-c,f,f-E])
 return dict(E=E,total=dict(zip(['rate','transport_rate','outgoing_energy_moment','neutrino_energy_change'],total.tolist())),controls=dict(outer=order,angle=angle,upper=upper,target_tail=tail),seconds=time.monotonic()-start)
