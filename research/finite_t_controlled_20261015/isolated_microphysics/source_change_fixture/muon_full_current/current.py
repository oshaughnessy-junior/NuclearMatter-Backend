"""Vacuum full V/A/weak-magnetism/pseudoscalar source-defined current.

Uses frozen vacuum_cc kinematics/gamma representation; no medium extension.
"""
from __future__ import annotations
from pathlib import Path
import sys,math
import numpy as np
from dataclasses import dataclass
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'muon_vacuum'))
import vacuum_cc as V

@dataclass(frozen=True)
class FormFactors:
    gv:float=1.
    ga:float=1.2723
    xi:float=3.706
    vector_mass:float=840.
    axial_mass:float=1000.
    pion_mass:float=139.57
    weak_magnetism:bool=True
    pseudoscalar:bool=True
    q2_dependence:bool=True

    def __post_init__(self):
        if not all(math.isfinite(x)for x in (self.gv,self.ga,self.xi,self.vector_mass,self.axial_mass,self.pion_mass))or min(self.vector_mass,self.axial_mass,self.pion_mass)<=0:raise ValueError('Finite couplings and positive pole masses required')
        if not all(isinstance(x,bool)for x in (self.weak_magnetism,self.pseudoscalar,self.q2_dependence)):raise ValueError('Explicit bool current switches required')

    def values(self,t,M):
        if not math.isfinite(t)or not math.isfinite(M)or M<=0:raise ValueError('Finite t and positive mass normalization required')
        if t>=min(self.pion_mass**2,self.vector_mass**2,self.axial_mass**2,4*M*M):raise ValueError('Outside declared sub-pole domain')
        # Exact Guo Eq27; xi=gamma_p-gamma_n-1.
        if self.q2_dependence:
            den=(1-t/(4*M*M))*(1-t/self.vector_mass**2)**2
            gv=self.gv*(1-(1+self.xi)*t/(4*M*M))/den
            ga=self.ga/(1-t/self.axial_mass**2)**2
            ff=self.xi/den
        else:gv,ga,ff=self.gv,self.ga,self.xi
        if self.pion_mass**2-t<=0:raise ValueError('Outside declared sub-pion-pole domain')
        gp=2*M*M*ga/(self.pion_mass**2-t)
        return gv,ga,ff if self.weak_magnetism else 0.,gp if self.pseudoscalar else 0.

def vertices(q,M,ff):
    gv,ga,f,gp=ff
    sigma=[[.5j*(V.G[u]@V.G[v]-V.G[v]@V.G[u])for v in range(4)]for u in range(4)]
    return dict(V=[gv*g for g in V.G],A=[-ga*g@V.G5 for g in V.G],
                F=[1j*f/(2*M)*sum(sigma[u][v]*V.METRIC[v]*q[v]for v in range(4))for u in range(4)],
                P=[-gp/M*V.G5*q[u]for u in range(4)])

def trace(channel,p,ff,*,decompose=False):
    q=p['pb']-p['pa'];M=(channel.initial_mass+channel.final_mass)/2
    vs=vertices(q,M,ff);k,pa,l,pb=(p[x]for x in ('k','pa','l','pb'))
    left=[g@(V.I-V.G5)for g in V.G]
    lepton=V.slash(l)+(-1 if channel.antineutrino else 1)*channel.lepton_mass*V.I
    L=np.array([[np.trace(lepton@left[u]@V.slash(k)@V.adjoint(left[v]))for v in range(4)]for u in range(4)])
    if channel.antineutrino:L=L.T
    a=V.slash(pa)+channel.initial_mass*V.I;b=V.slash(pb)+channel.final_mass*V.I
    def contraction(j,k):
        H=np.array([[.5*np.trace(b@j[u]@a@V.adjoint(k[v]))for v in range(4)]for u in range(4)])
        z=V.GF**2*V.VUD**2/2*np.sum(V.METRIC[:,None]*V.METRIC[None,:]*L*H)
        if abs(z.imag)>1e-10*max(abs(z.real),1e-300):raise ArithmeticError('Nonreal trace')
        return float(z.real)
    if decompose:
        terms={}
        for u,v in [('V','V'),('A','A'),('F','F'),('P','P'),('V','A'),('V','F'),('A','F'),('A','P'),('V','P'),('F','P')]:
            terms[u+v]=contraction(vs[u],vs[v])+(contraction(vs[v],vs[u])if u!=v else 0.)
        return terms
    allv=[sum(vs[k][u]for k in vs)for u in range(4)]
    value=contraction(allv,allv)
    if value<0:raise ArithmeticError('Negative full amplitude')
    return value

def guo_invariants(channel,p,ff):
    """Independent published Eq47a-h, vacuum q*=q; no calibration factors."""
    v,a,f,g=ff;k,n,l,b=(p[x]for x in ('k','pa','l','pb'));q=b-n
    mn,mp=channel.initial_mass,channel.final_mass;M=(mn+mp)/2
    kn,kb,ln,lb,kl,nb=(V.dot(x,y)for x,y in [(k,n),(k,b),(l,n),(l,b),(k,l),(n,b)])
    kq,lq,nq,bq,t=(V.dot(x,y)for x,y in [(k,q),(l,q),(n,q),(b,q),(q,q)])
    pref=16*V.GF**2*V.VUD**2;s=-1 if channel.antineutrino else 1
    return dict(VV=pref*v*v*(kn*lb+kb*ln-mn*mp*kl),
      VA=s*2*pref*v*a*(kn*lb-kb*ln),AA=pref*a*a*(kn*lb+kb*ln+mn*mp*kl),
      VF=pref*v*f/(2*M)*((kn*mp-kb*mn)*lq+(ln*mp-lb*mn)*kq+(nq*mp-bq*mn)*kl),
      AF=s*pref*a*f/M*((kn*mp+kb*mn)*lq-(ln*mp+lb*mn)*kq),
      FF=pref*f*f/(8*M*M)*(2*(ln*bq+lb*nq)*kq+2*(kn*bq+kb*nq)*lq-2*kq*lq*nb+t*(kl*nb-2*kn*lb-2*kb*ln)-mn*mp*(kl*t+2*kq*lq)),
      AP=pref*a*g/M*(kq*(ln*mp-lb*mn)+lq*(kn*mp-kb*mn)+kl*(bq*mn-nq*mp)),
      PP=pref*g*g/(2*M*M)*(2*kq*lq-t*kl)*(nb-mn*mp))

def differential(channel,E,c,model=FormFactors(),*,method='trace'):
    if method not in('trace','source'):raise ValueError('Unknown current amplitude method')
    if channel.gv!=model.gv or channel.ga!=model.ga:raise ValueError('Channel and current benchmark gv/ga must match explicitly')
    p=V.momenta(channel,E,c)
    if p is None:return 0.
    t=V.dot(p['k']-p['l'],p['k']-p['l']);M=(channel.initial_mass+channel.final_mass)/2;ff=model.values(t,M)
    amp=trace(channel,p,ff)if method=='trace'else sum(guo_invariants(channel,p,ff).values())
    return amp/(32*math.pi*p['s'])*p['pf']/p['pi']*V.HBARC**2*1e-26

def total(channel,E,model=FormFactors(),*,order=32,method='trace'):
    if isinstance(order,bool)or not isinstance(order,int)or order<4:raise ValueError('Integer order>=4 required')
    x,w=leggauss(order)
    return float(sum(wi*differential(channel,E,float(xi),model,method=method)for xi,wi in zip(x,w)))
