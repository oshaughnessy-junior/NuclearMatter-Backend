"""Exact vacuum 2-body CC with point V-A nucleons and explicit lepton mass.

Metric +---; spin-average initial target only; one active incident helicity.
No weak magnetism, pseudoscalar, q² form factors or radiative corrections.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from numpy.polynomial.legendre import leggauss

GF=1.1663785e-11 # MeV^-2
VUD=.974       # benchmark convention, not a fitted CKM estimate
HBARC=197.3269804 # MeV fm
MN=939.56542052
MP=938.27208816
ME=.51099895069
MMU=105.6583755
METRIC=np.array([1.,-1.,-1.,-1.])
I=np.eye(4,dtype=complex)
Z=np.zeros((2,2),complex);I2=np.eye(2)
paulis=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.array([[1,0],[0,-1]],complex)]
G=[np.diag([1.,1.,-1.,-1.]).astype(complex)]
G.extend(np.block([[Z,s],[-s,Z]])for s in paulis)
G5=1j*G[0]@G[1]@G[2]@G[3]

@dataclass(frozen=True)
class Channel:
    initial_mass:float
    final_mass:float
    lepton_mass:float
    antineutrino:bool=False
    gv:float=1.
    ga:float=1.2723

    def __post_init__(self):
        vals=(self.initial_mass,self.final_mass,self.lepton_mass,self.gv,self.ga)
        if not all(math.isfinite(x) for x in vals)or min(vals[:2])<=0 or self.lepton_mass<0:raise ValueError('Finite positive nucleon masses, nonnegative lepton mass required')
        if not isinstance(self.antineutrino,bool):raise ValueError('Explicit neutrino/antineutrino bool required')

    @property
    def threshold(self):
        # Factored invariant difference avoids subtracting two large mass².
        a,b,m=self.initial_mass,self.final_mass,self.lepton_mass
        return max(0.,(b+m-a)*(b+m+a)/(2*a))

def dot(a,b):return float(np.dot(METRIC*a,b))
def slash(p):return sum(METRIC[i]*p[i]*G[i] for i in range(4))
def adjoint(g):return G[0]@g.conjugate().T@G[0]

def momenta(channel,E,cosine):
    if not math.isfinite(E)or E<=0 or not math.isfinite(cosine)or not -1<=cosine<=1:raise ValueError('E>0 and cosine in[-1,1] required')
    if E<=channel.threshold:return None
    a,b,m=channel.initial_mass,channel.final_mass,channel.lepton_mass
    s=a*a+2*a*E;root=math.sqrt(s);pi=a*E/root
    # Exact threshold distance expressed without cancellation in s-(b+m)².
    first=2*a*(E-channel.threshold) if channel.threshold>0 else (a-b-m)*(a+b+m)+2*a*E
    second=(a-b+m)*(a+b-m)+2*a*E
    pf=math.sqrt(first*second)/(2*root)
    El=math.hypot(pf,m)
    # sqrt(pi²+a²) represented by root-pi, enforcing exact CM conservation.
    k=np.array([pi,0.,0.,pi]);pa=np.array([root-pi,0.,0.,-pi])
    transverse=pf*math.sqrt(max(0.,1-cosine*cosine))
    l=np.array([El,transverse,0.,pf*cosine]);pb=np.array([root-El,-transverse,0.,-pf*cosine])
    return dict(k=k,pa=pa,l=l,pb=pb,s=s,pi=pi,pf=pf)

def trace_amplitude(channel,p):
    k,pa,l,pb=(p[x] for x in ('k','pa','l','pb'))
    left=[g@(I-G5)for g in G]
    # Sum of both massless spin states is safe: inactive state is killed by V-A.
    # Thus there is no incoming-neutrino 1/2 average.
    lepton=slash(l)+(-1 if channel.antineutrino else 1)*channel.lepton_mass*I
    L=np.array([[np.trace(lepton@left[u]@slash(k)@adjoint(left[v]))for v in range(4)]for u in range(4)])
    if channel.antineutrino:L=L.T
    J=[g@(channel.gv*I-channel.ga*G5)for g in G]
    a=slash(pa)+channel.initial_mass*I;b=slash(pb)+channel.final_mass*I
    H=np.array([[.5*np.trace(b@J[u]@a@adjoint(J[v]))for v in range(4)]for u in range(4)])
    value=GF**2*VUD**2/2*np.sum(METRIC[:,None]*METRIC[None,:]*L*H)
    if abs(value.imag)>1e-10*max(abs(value.real),1e-300):raise ArithmeticError('Nonreal trace contraction')
    if value.real<0:raise ArithmeticError('Negative squared amplitude')
    return float(value.real)

def invariant_amplitude(channel,p):
    """Guo2020 Eq47a-c vacuum reduction; separately coded scalar products."""
    k,a,l,b=(p[x]for x in ('k','pa','l','pb'))
    X,Y=dot(k,a)*dot(l,b),dot(k,b)*dot(l,a)
    z=channel.initial_mass*channel.final_mass*dot(k,l)
    v,g=channel.gv,channel.ga;sign=-1 if channel.antineutrino else 1
    return 16*GF**2*VUD**2*((v*v+g*g)*(X+Y)+sign*2*v*g*(X-Y)-(v*v-g*g)*z)

def differential(channel,E,cosine,*,method='trace'):
    p=momenta(channel,E,cosine)
    if p is None:return 0.
    if method not in ('trace','invariant'):raise ValueError('Unknown amplitude method')
    amp=(trace_amplitude if method=='trace' else invariant_amplitude)(channel,p)
    # d sigma/d cos thetaCM; azimuth already integrated; MeV^-2 -> cm².
    return amp/(32*math.pi*p['s'])*p['pf']/p['pi']*HBARC**2*1e-26

def total(channel,E,*,order=32,method='trace'):
    if isinstance(order,bool)or not isinstance(order,int)or order<4:raise ValueError('Integer order>=4 required')
    x,w=leggauss(order)
    return float(sum(v*differential(channel,E,float(c),method=method)for c,v in zip(x,w)))

def static_total(channel,E):
    """Heavy-target limit only; not a finite-mass vacuum threshold rule."""
    El=E+channel.initial_mass-channel.final_mass
    if El<=channel.lepton_mass:return 0.
    return GF**2*VUD**2/math.pi*(channel.gv**2+3*channel.ga**2)*El*math.sqrt((El-channel.lepton_mass)*(El+channel.lepton_mass))*HBARC**2*1e-26
