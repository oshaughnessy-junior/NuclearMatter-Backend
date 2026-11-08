"""Local eigenfunction-overlap continuation across a resolved parameter step."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from .pencil import ModeError


def match_branches(previous,current,*,weight=None,minimum_overlap=.5,frequency_weight=.05):
    """Return current column indices in previous branch order and overlaps.

    Users supply compatible physical basis coordinates and a positive metric.
    Gauge/constraint projection belongs in the physical assembler. Close
    degeneracies require subspace continuation rather than individual labels.
    """
    if previous.right.shape!=current.right.shape or not 0<=minimum_overlap<=1 or not np.isfinite(frequency_weight) or frequency_weight<0:
        raise ModeError('Branch count/basis changed or invalid overlap threshold')
    n=previous.right.shape[0]
    w=np.eye(n) if weight is None else np.asarray(weight)
    if w.shape!=(n,n) or not np.all(np.isfinite(w)) or not np.allclose(w,w.conj().T):
        raise ModeError('Branch metric must be finite Hermitian positive definite')
    try: np.linalg.cholesky(w)
    except np.linalg.LinAlgError as error: raise ModeError('Nonpositive branch metric') from error
    x,y=previous.right,current.right
    norms_x=np.real(np.sum(x.conj()*(w@x),axis=0))
    norms_y=np.real(np.sum(y.conj()*(w@y),axis=0))
    overlaps=abs(x.conj().T@w@y)**2/(norms_x[:,None]*norms_y[None,:])
    scale=max(np.max(abs(previous.sigma),initial=0.),np.max(abs(current.sigma),initial=0.),np.finfo(float).tiny)
    distance=abs(previous.sigma[:,None]-current.sigma[None,:])/scale
    cost=1-np.minimum(overlaps,1.)+frequency_weight*distance
    if cost.shape[1]>1 and np.any(np.diff(np.sort(cost,axis=1)[:,:2],axis=1)<1e-12):
        raise ModeError('Ambiguous mode matching: refine the sequence or track a subspace')
    rows,cols=linear_sum_assignment(cost)
    chosen=overlaps[rows,cols]
    if np.any(chosen<minimum_overlap):
        raise ModeError('Unresolved continuation step: refine the sequence/subspace')
    return cols,chosen
