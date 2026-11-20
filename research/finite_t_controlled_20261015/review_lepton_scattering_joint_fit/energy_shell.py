"""Independent target-energy shell/analytic-azimuth moments, fixed e± bath."""
import math,time
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
T=2.;MU=.17446333200826397;M=.51099895069;GF=1.1663785e-11;HBAR=6.582119569e-22

def differential(E,F,c,pos,tail=80.):
    w=E-F;a=2*E*F*(1-c);q=math.sqrt(w*w+a)
    lo=(-w+q*math.sqrt(1+4*M*M/a))/2
    mu=-MU if pos else MU;hi=max(lo,mu,mu-w)+tail*T
    kz=(E*E+q*q-F*F)/(2*q);kx2=max(0.,E*E-kz*kz);h=a/2
    L,R=(.96258+.5)**2,(.96258-.5)**2
    if pos:L,R=R,L
    def integrand(e):
        pz=(2*e*w-a)/(2*q);r2=max(0.,e*e-M*M-pz*pz)
        u=E*e-kz*pz;az=kx2*r2/2
        amplitude=L*(u*u+az)+R*((u-h)**2+az)-M*M*(.96258**2-.5**2)*h
        return amplitude*expit((mu-e)/T)*expit((e+w-mu)/T)
    pts=sorted(set([lo,hi]+[x for mid in [mu,mu-w] for d in [-8,-3,0,3,8] if lo<(x:=mid+d*T)<hi]))
    value=sum(quad(integrand,l,h,epsabs=2e-9,epsrel=2e-11,limit=100)[0] for l,h in zip(pts[:-1],pts[1:]))
    return F/(64*math.pi**3*E*q*HBAR)*16*GF*GF*value

def integrate(E,outer=18,angle=30,upper=160.,tail=80.):
    start=time.monotonic();xx,ww=leggauss(outer);zz,zw=leggauss(angle);vals=np.zeros(4)
    # Independent target-energy integral; outer z=sin(theta/2) resolves q→0.
    for z,wz in zip((zz+1)/2,zw/2):
        c=1-2*z*z;width=2*E*z
        edges=sorted(set([0.,1.5,4.,E,15.,30.,60.,upper]+[E+h*width for h in [-5,-2.5,-1,-.35,-.1,.1,.35,1,2.5,5] if 0<E+h*width<upper]))
        edges=[a for a in edges if a<=upper]
        for l,h in zip(edges[:-1],edges[1:]):
            for F,wf in zip((l+h)/2+xx*(h-l)/2,ww*(h-l)/2):
                if time.monotonic()-start>110:raise TimeoutError('independent bounded shell integral')
                d=sum(differential(E,float(F),float(c),pos,tail) for pos in [False,True])
                vals+=d*wf*wz*4*z*np.array([1.,1-c,F,F-E])
    return dict(E=E,total=dict(zip(['rate','transport_rate','outgoing_energy_moment','neutrino_energy_change'],vals.tolist())),controls=dict(outer=outer,angle=angle,upper=upper,tail=tail),seconds=time.monotonic()-start)
