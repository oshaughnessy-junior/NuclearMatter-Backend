"""Independent analytic-azimuth transverse-shell directional edge reference."""
import math
from scipy.integrate import quad
from scipy.special import expit
M=.51099895069;GF=1.1663785e-11;HBAR=6.582119569e-22

def differential(E,F,c,T,mu,pos):
 mut=-mu if pos else mu;v=.96258;a=.5;om=E-F;Q=math.sqrt(om*om+2*E*F*(1-c));s=2*E*F*(1-c);lo=(-om+Q*math.sqrt(1+4*M*M/s))/2;hi=max(lo,mut,mut-om)+90*T;kz=(E*E+Q*Q-F*F)/(2*Q);kx2=max(0.,E*E-kz*kz);h=s/2;L,R=(v+a)**2,(v-a)**2
 if pos:L,R=R,L
 pz=lambda e:(2*e*om-s)/(2*Q)
 rmax=math.sqrt(max(0.,hi*hi-M*M-pz(hi)**2))
 def f(r):
  rr=math.sqrt(1+4*(M*M+r*r)/s);e=(-om+Q*rr)/2;de=2*Q*r/(s*rr);u=E*e-kz*pz(e);z=kx2*r*r/2;am=L*(u*u+z)+R*((u-h)**2+z)-M*M*(v*v-a*a)*h
  return de*am*expit((mut-e)/T)*expit((e+om-mut)/T)
 val=quad(f,0,rmax,epsabs=1e-8,epsrel=5e-10,limit=120)[0]
 return F/(64*math.pi**3*E*Q*HBAR)*16*GF*GF*val

def edge(E,F,mu,mup,T,mue):
 def phi(ph):
  c=mu*mup+math.sqrt((1-mu*mu)*(1-mup*mup))*math.cos(ph)
  return sum(differential(E,F,c,T,mue,pos)for pos in[False,True])/(2*math.pi)
 return quad(phi,0,2*math.pi,epsabs=2e-10,epsrel=2e-8,limit=120)[0]
