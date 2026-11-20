"""Off-pole common-mass NR external-leg OPE radiation algebra, NOT a MU rate.

Outgoing weak four-momentum (omega,Q). Operators map labelled incoming to
outgoing nucleon spin/isospin states. No lepton vertex, GF, phase measure or MEC.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import math
import numpy as np

_path=Path(__file__).resolve().parents[1]/'ope_nucleon_amplitudes'/'ope.py'
_spec=importlib.util.spec_from_file_location('ope_weak_pinned_strong',_path)
ope=importlib.util.module_from_spec(_spec)
sys.modules[_spec.name]=ope
_spec.loader.exec_module(ope)
I=np.eye(2,dtype=complex)
LOWER=np.array([[0,0],[1,0]],complex)  # n->p in (n,p)

def rho(line,raising=False):
    if line not in (0,1):raise ValueError('Line must be 0 or 1')
    t=LOWER.conj().T if raising else LOWER
    return np.kron(np.eye(4),np.kron(t,I) if line==0 else np.kron(I,t))

def sigma(line,axis):
    return np.kron(np.kron(ope.PAULI[axis],I) if line==0 else np.kron(I,ope.PAULI[axis]),np.eye(4))

def validate(p,omega,Q,model,pole_gap):
    if np.iscomplexobj(p) or np.iscomplexobj(Q):raise ValueError('Real momenta required')
    p=np.asarray(p,float);Q=ope.vector(Q)
    if p.shape!=(4,3) or not np.all(np.isfinite(p)):raise ValueError('Four finite three-momenta required')
    if not math.isfinite(omega) or not math.isfinite(pole_gap) or pole_gap<=0:raise ValueError('Finite omega and positive finite exclusion gap required')
    scale=max(1.,float(np.max(abs(p))),abs(omega))
    if np.max(abs(p[0]+p[1]-p[2]-p[3]-Q))>1e-10*scale:raise ValueError('Outgoing weak momentum not conserved')
    with np.errstate(over='ignore',invalid='ignore'):
        ek=np.sum(p*p,axis=1)/(2*model.common_mass)
        qsq=Q@Q
    if not np.all(np.isfinite(ek)) or not math.isfinite(qsq):raise ValueError('Nonrepresentable kinetic invariants')
    if abs(ek[0]+ek[1]-ek[2]-ek[3]-omega)>1e-10*max(1.,float(np.max(ek)),abs(omega)):raise ValueError('Common-mass NR energy not conserved')
    d=np.array([-omega+(p[i]@Q)/(model.common_mass)-(Q@Q)/(2*model.common_mass) for i in (0,1)]+
               [omega-(p[i]@Q)/(model.common_mass)-(Q@Q)/(2*model.common_mass) for i in (2,3)])
    if not np.all(np.isfinite(d)):raise ValueError('Nonrepresentable propagator denominator')
    if np.min(abs(d))<=pole_gap:raise ValueError('External propagator excluded: |D| <= pole_gap; no width supplied')
    return p,Q,d

def direct(p,omega,Q,component,model=ope.Model(),pole_gap=.05,gv=1.,ga=1.26,raising=False):
    """Four labelled leg insertions, retained coherently; output units MeV^-3.

    V0 is charge, Vx/y/z is NR convection current; Ax/y/z is minus the
    leading spatial axial current (V-A convention). No axial charge provided.
    Gap REJECTS a numerical domain: it is not an imaginary regulator or width.
    """
    p,Q,d=validate(p,omega,Q,model,pole_gap)
    if not isinstance(raising,(bool,np.bool_)):raise ValueError('raising must be Boolean')
    if component not in ('V0','Vx','Vy','Vz','Ax','Ay','Az'):raise ValueError('Unsupported current component')
    if not all(math.isfinite(x) for x in (gv,ga)):raise ValueError('Finite weak couplings required')
    q1=p[2]-p[0];q2=p[3]-p[1]
    vs=(ope.direct(q2,model),ope.direct(q1,model))
    out=np.zeros((16,16),complex)
    for line in (0,1):
        r=rho(line,raising)
        if component=='V0':ji=jo=gv*r
        elif component[0]=='V':
            axis='xyz'.index(component[1])
            ji=gv*(2*p[line,axis]-Q[axis])/(2*model.common_mass)*r
            jo=gv*(2*p[line+2,axis]+Q[axis])/(2*model.common_mass)*r
        else:ji=jo=-ga*sigma(line,'xyz'.index(component[1]))@r
        out+=vs[line]@ji/d[line]+jo@vs[line]/d[line+2]
    return out

def amplitude(p,omega,Q,component,**kwargs):
    """Labelled final antisymmetrization B(34)-P B(43), no sqrt2 or spin average."""
    p=np.asarray(p,float)
    first=direct(p,omega,Q,component,**kwargs)
    swapped=p[[0,1,3,2]]
    return first-ope.EXCHANGE@direct(swapped,omega,Q,component,**kwargs)

def vector_defect_direct(p,omega,Q,model=ope.Model(),pole_gap=.05,gv=1.,raising=False):
    """Independent denominator-free RHS of omega*B0-Q.Bv, NOT a conserved current."""
    p,Q,_=validate(p,omega,Q,model,pole_gap)
    v=(ope.direct(p[3]-p[1],model),ope.direct(p[2]-p[0],model))
    return gv*sum(rho(i,raising)@v[i]-v[i]@rho(i,raising) for i in (0,1))

def channel(a,initial,final):
    if any(c not in ope.ISO for c in initial+final) or len(initial)!=2 or len(final)!=2:raise ValueError('Explicit two-nucleon channel required')
    i=2*ope.ISO[initial[0]]+ope.ISO[initial[1]];f=2*ope.ISO[final[0]]+ope.ISO[final[1]]
    return a[np.ix_(np.arange(4)*4+f,np.arange(4)*4+i)]

def emission_point(pmag,omega,Q,direction,model=ope.Model()):
    """Common-mass free NR CM input -> recoiling pair + outgoing weak momentum."""
    if np.iscomplexobj(Q) or np.iscomplexobj(direction):raise ValueError('Real momenta required')
    Q=ope.vector(Q);u=ope.vector(direction)
    if not math.isfinite(pmag) or pmag<0 or not math.isfinite(omega) or np.linalg.norm(u)==0:raise ValueError('Finite physical point inputs required')
    r2=pmag**2-model.common_mass*omega-(Q@Q)/4
    if r2<0:raise ValueError('No final pair kinetic support')
    r=math.sqrt(r2)*u/np.linalg.norm(u)
    return np.array([[0,0,pmag],[0,0,-pmag],-Q/2+r,-Q/2-r])
