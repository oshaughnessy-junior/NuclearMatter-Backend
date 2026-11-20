"""Pinned-O2scl-equation reduction, NOT execution of O2scl.

Momentum-space Gauss-Legendre moments and nested scalar/density Brent roots.
No internal GM1 functions or values enter this solver. Units: fm powers.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from numpy.polynomial.legendre import leggauss

HC=197.3269804
M=939./HC
CS, CW, CR=11.785,7.148,4.410
B,C=.002947,-.001070

def solve(nB=.16,T_MeV=10.,Yp=.1,*,order=128,tail=50.):
    if not all(math.isfinite(x) for x in (nB,T_MeV,Yp,tail)) or nB<=0 or T_MeV<=0 or not 0<Yp<1 or tail<=0:
        raise ValueError('finite positive nB,T,tail and interior Yp required')
    if isinstance(order,bool) or not isinstance(order,int) or order<16: raise ValueError('integer order>=16 required')
    T=T_MeV/HC; z,w=leggauss(order)
    def moments(m,nu):
        # Two momentum panels: resolve the nominal kinetic Fermi surface.
        Eend=max(m,abs(nu))+tail*T
        pend=math.sqrt((Eend-m)*(Eend+m))
        pf=math.sqrt(max(0.,(abs(nu)-m)*(abs(nu)+m)))
        out=np.zeros(6)
        edges=[0.,pf,pend] if pf>0 else [0.,pend]
        for a,b in zip(edges[:-1],edges[1:]):
            p=a+(b-a)*(z+1)/2; v=w*(b-a)/2*p*p/math.pi**2
            e=np.hypot(p,m); x=(e-nu)/T; y=(e+nu)/T
            f=expit(-x); fb=expit(-y)
            h=lambda a: np.logaddexp(0.,-abs(a))+abs(a)*expit(-abs(a))
            out+=np.array([np.dot(v,f-fb),np.dot(v,m/e*(f+fb)),np.dot(v,e*(f+fb)),np.dot(v,p*p/(3*e)*(f+fb)),np.dot(v,h(x)+h(y)),np.dot(v,fb)])
        return out
    def at_mass(m,n):
        nu=brentq(lambda nu:moments(m,nu)[0]/n-1,0.,max(2*M,m+100*T),xtol=4e-14,rtol=2e-14)
        return nu,moments(m,nu)
    def scalar(m,full=False):
        nun,gn=at_mass(m,nB*(1-Yp)); nup,gp=at_mass(m,nB*Yp)
        phi=M-m
        # Source: eos_had_rmf::calc_eq_temp_p, divide field f1 by gs.
        residual=phi/CS+B*M*phi**2+C*phi**3-gn[1]-gp[1]
        return (nun,nup,gn,gp,residual) if full else residual
    # Scan first upcross in Phi without importing cold/internal solutions.
    previous=(0.,scalar(M))
    for phi in np.linspace(0.,.985*M,101)[1:]:
        here=scalar(M-phi)
        if previous[1]<0<=here:
            root=brentq(lambda phi:scalar(M-phi),previous[0],phi,xtol=3e-14,rtol=2e-14)
            break
        previous=(phi,here)
    else: raise ArithmeticError('No first scalar upcross')
    m=M-root; nun,nup,gn,gp,res=scalar(m,True)
    W=CW*nB; R=CR*nB*(2*Yp-1)/2
    U=root**2/(2*CS)+B*M*root**3/3+C*root**4/4
    V=W*W/(2*CW)+R*R/(2*CR)
    e=(gn[2]+gp[2]+U+V)*HC; P=(gn[3]+gp[3]-U+V)*HC
    sn=gn[4]+gp[4]
    return dict(nB=nB,T_MeV=T_MeV,Yp=Yp,nn=gn[0],np=gp[0],phi_mev=root*HC,omega_mev=W*HC,rho_mev=R*HC,dirac_mass_mev=m*HC,mu_n_mev=(nun+W-R/2)*HC,mu_p_mev=(nup+W+R/2)*HC,energy_density=e,pressure=P,entropy_density=sn,free_energy_density=e-T_MeV*sn,scalar_density=gn[1]+gp[1],antiparticle_density=gn[5]+gp[5],normalized_scalar_residual=res/nB,normalized_n_residual=gn[0]/(nB*(1-Yp))-1,normalized_p_residual=gp[0]/(nB*Yp)-1,order=order,tail=tail)
