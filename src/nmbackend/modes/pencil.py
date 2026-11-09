"""Quadratic eigenproblems for mode amplitudes exp(i*(sigma*t + m*phi)).

Motivated by Gao & Shibata (arXiv:2611.03986), Eq. 3; independently
implemented numerical infrastructure, not the full Einstein-fluid assembler.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eig


class ModeError(ValueError):
    """An unresolved, singular or ill-conditioned mode cannot be admitted."""


@dataclass(frozen=True)
class Pencil:
    k0: np.ndarray
    k1: np.ndarray
    k2: np.ndarray

    def __post_init__(self):
        arrays = [np.array(a, dtype=np.complex128, copy=True) for a in (self.k0,self.k1,self.k2)]
        n = arrays[0].shape[0] if arrays[0].ndim == 2 else 0
        if not n or any(a.shape != (n,n) or not np.all(np.isfinite(a)) for a in arrays):
            raise ModeError('Require three finite square matrices with identical nonzero shape')
        for name,a in zip(('k0','k1','k2'),arrays):
            a.setflags(write=False)
            object.__setattr__(self,name,a)

    def at(self,sigma):
        return self.k0+sigma*self.k1+sigma*sigma*self.k2

    def derivative(self,sigma):
        return self.k1+2*sigma*self.k2


@dataclass(frozen=True)
class Spectrum:
    sigma: np.ndarray
    right: np.ndarray
    left: np.ndarray
    residual: np.ndarray
    left_residual: np.ndarray
    derivative_condition: np.ndarray
    infinite_roots: int


def solve(pencil,*,residual_tolerance=1e-8,max_dimension=256):
    """Generalized companion solve, retaining a count of infinite roots.

    K2 may be singular. No inversion of K2 or fake constraint regularization
    is performed. This dense kernel is intended for validation-sized systems;
    full spectral GR needs a separately qualified sparse assembly/solver.
    """
    if not 0 < residual_tolerance < 1 or max_dimension < 1:
        raise ModeError('Invalid residual tolerance or dense dimension limit')
    n = len(pencil.k0)
    if n > max_dimension:
        raise ModeError('Dense mode dimension limit exceeded; use a qualified sparse solver')
    zero=np.zeros((n,n),complex)
    eye=np.eye(n,dtype=complex)
    a=np.block([[zero,eye],[-pencil.k0,-pencil.k1]])
    b=np.block([[eye,zero],[zero,pencil.k2]])
    ab,left,right=eig(a,b,left=True,right=True,homogeneous_eigvals=True)
    alpha,beta=ab
    if np.any((alpha==0)&(beta==0)):
        raise ModeError('Singular companion pencil has indeterminate 0/0 roots')
    finite=beta!=0
    infinite=int(np.sum(~finite))
    sigma=alpha[finite]/beta[finite]
    if not np.all(np.isfinite(sigma)):
        raise ModeError('Finite-root overflow; rescale the pencil')
    x=right[:n,finite].copy()
    y=left[n:,finite].copy()
    if np.any(np.linalg.norm(x,axis=0)==0) or np.any(np.linalg.norm(y,axis=0)==0):
        raise ModeError('Companion eigenvector has no physical polynomial component')
    x/=np.linalg.norm(x,axis=0)
    y/=np.linalg.norm(y,axis=0)
    residual=[]
    left_residual=[]
    condition=[]
    norms=[np.linalg.norm(k,2) for k in (pencil.k0,pencil.k1,pencil.k2)]
    for j,s in enumerate(sigma):
        p=pencil.at(s)
        scale=norms[0]+abs(s)*norms[1]+abs(s*s)*norms[2]
        scale=max(scale,np.finfo(float).tiny)
        residual.append(np.linalg.norm(p@x[:,j])/scale)
        left_residual.append(np.linalg.norm(y[:,j].conj()@p)/scale)
        slope_scale=norms[1]+2*abs(s)*norms[2]
        denom=np.vdot(y[:,j],pencil.derivative(s)@x[:,j])
        condition.append(abs(denom)/max(slope_scale,np.finfo(float).tiny))
    if max([*residual,*left_residual],default=0.)>residual_tolerance:
        raise ModeError('Polynomial left/right residual exceeds acceptance tolerance')
    order=np.lexsort((sigma.imag,sigma.real))
    return Spectrum(sigma[order],x[:,order],y[:,order],np.asarray(residual)[order],
                    np.asarray(left_residual)[order],np.asarray(condition)[order],infinite)


def select_simple(pencil,reference,*,condition_tolerance=1e-7,gap_tolerance=1e-9):
    """Select a local simple root; reference is an anchor, not a global mode label."""
    if not np.isfinite(reference):
        raise ModeError('Require a finite root anchor')
    modes=solve(pencil)
    if not len(modes.sigma):
        raise ModeError('No finite eigenvalues')
    index=int(np.argmin(abs(modes.sigma-reference)))
    if modes.derivative_condition[index] < condition_tolerance:
        raise ModeError('Mode derivative is ill-conditioned or at an exceptional point')
    separation=np.delete(abs(modes.sigma-modes.sigma[index]),index)
    if len(separation) and min(separation)<=gap_tolerance*max(1.,abs(modes.sigma[index])):
        raise ModeError('Repeated/unresolved root; differentiate a subspace instead')
    return modes,index


def directional_derivative(pencil,modes,index,dk0,dk1,dk2):
    """Implicit simple-root derivative using physical left and right vectors."""
    if modes.derivative_condition[index]<1e-7:
        raise ModeError('Ill-conditioned eigenvalue derivative')
    delta=Pencil(dk0,dk1,dk2)
    s=modes.sigma[index]
    x,y=modes.right[:,index],modes.left[:,index]
    return -np.vdot(y,delta.at(s)@x)/np.vdot(y,pencil.derivative(s)@x)
