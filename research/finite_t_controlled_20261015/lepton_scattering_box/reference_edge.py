"""Independent r-shell/analytic-azimuth edge quadrature (no upstream drivers).

Derived from review_neutrino_lepton's stated alternate reduction; literal local
function, not importing its unguarded probe. Shared weak coefficients explicit.
"""
import math
from scipy.integrate import quad
from scipy.special import expit
M=.51099895069;GF=1.1663785e-11;HBAR=6.582119569e-22

def differential(T,mu_e,E,F,c,cv,ca,pos=False,tail=80.):
 om=E-F;a=2*E*F*(1-c);Q=math.sqrt(om*om+a);emin=(-om+Q*math.sqrt(1+4*M*M/a))/2;mu=-mu_e if pos else mu_e;upper=max(emin,mu,mu-om)+tail*T;kz=(E*E+Q*Q-F*F)/(2*Q);kx2=max(0.,E*E-kz*kz);h=E*F*(1-c);L,R=(cv+ca)**2,(cv-ca)**2
 if pos:L,R=R,L
 pz=lambda e:(2*e*om-a)/(2*Q)
 rmax=math.sqrt(max(0.,upper*upper-M*M-pz(upper)**2))
 def integrand(r):
  root=math.sqrt(1+4*(M*M+r*r)/a);et=(-om+Q*root)/2;der=2*Q*r/(a*root);u=E*et-kz*pz(et);vv=kx2*r*r/2
  amp=L*(u*u+vv)+R*((u-h)**2+vv)-M*M*(cv*cv-ca*ca)*h
  return der*amp*expit((mu-et)/T)*expit((et+om-mu)/T)
 v=quad(integrand,0,rmax,epsabs=2e-9,epsrel=2e-10,limit=120)[0]
 return F/(64*math.pi**3*E*Q*HBAR)*16*GF*GF*v

def edge(T,mu,E,F,cv,ca):
 # Independent q-outer coordinate, dc=-q dq/(EF). Neither z nor
 # finite target-energy/phi Gaussian rules from the grid builder are used.
 lo=abs(E-F);hi=E+F
 def integrand(q):
  c=(E*E+F*F-q*q)/(2*E*F)
  if not -1<c<1:return 0.
  return q/(E*F)*sum(differential(T,mu,E,F,c,cv,ca,pos)for pos in[False,True])
 return quad(integrand,lo,hi,epsabs=1e-9,epsrel=2e-9,limit=160)[0]
