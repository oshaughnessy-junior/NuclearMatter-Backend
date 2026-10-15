"""Normalized degenerate five-fermion ENERGY-ONLY kernel; no interaction rate.

All x=(epsilon-mu)/T are linearized excitation energies; angular/interaction
prefactor is intentionally NOT supplied. Not arbitrary-degeneracy MUrca.
"""
import math
from scipy.integrate import quad
from scipy.special import expit,zeta

def f(x):return float(expit(-x))
def J2(x):
 if not math.isfinite(x):raise ValueError('Finite dimensionless energy required')
 if abs(x)<1e-5:return 1-x/2+x*x/12-x**4/720
 if x>0:
  u=math.exp(-x);return x*u/(-math.expm1(-x))
 return x/math.expm1(x)
def J3(x):return (x*x+math.pi**2)*f(x)/2

def J5(x):
 if not math.isfinite(x):raise ValueError('Finite dimensionless energy required')
 return (x**4+10*math.pi**2*x*x+9*math.pi**4)*f(x)/24

def convolution3(x,limit=90.):
 # Original f*f*f convolution reduced by independently checked f*f.
 return quad(lambda t:f(t)*J2(x-t),-limit,limit,epsabs=2e-12,epsrel=2e-11,points=sorted(set(v for v in [0.,x]if -limit<v<limit)),limit=200)[0]
def convolution5(x,limit=90.):
 return quad(lambda t:J2(t)*J3(x-t),-limit,limit,epsabs=2e-11,epsrel=2e-11,points=sorted(set(v for v in [0.,x]if -limit<v<limit)),limit=200)[0]

def moments(affinity=0.):
 # beta emission energy: sum five excitation variables = y-affinity.
 # Opposite charged-lepton capture branch in this Fermi-surface reduction
 # corresponds to affinity->-affinity. These are NOT microscopic inverse
 # absorption rates with finite incident-neutrino occupation.
 if not math.isfinite(affinity)or abs(affinity)>10:raise ValueError('Selected |delta_mu/T|<=10 only')
 out={}
 for power,name in [(2,'number'),(3,'energy')]:
  out[name]=quad(lambda y:y**power*J5(y-affinity),0,math.inf,epsabs=1e-8,epsrel=2e-11,limit=200)[0]
 return out

def equilibrium_analytic():
 I=lambda p:(1-2**(-p))*math.gamma(p+1)*float(zeta(p+1))
 return dict(number=(I(6)+10*math.pi**2*I(4)+9*math.pi**4*I(2))/24,energy=11513*math.pi**8/120960)
