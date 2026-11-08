"""Local implicit QEP eigenvalue JVP; JAX transposes it for reverse AD."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from .pencil import Pencil,select_simple


def _packed_callback(k0,k1,k2,reference):
    pencil=Pencil(k0,k1,k2)
    spectrum,index=select_simple(pencil,complex(reference))
    return np.concatenate([[spectrum.sigma[index]],spectrum.right[:,index],spectrum.left[:,index]]).astype(np.complex128)


def _pack(k0,k1,k2,reference):
    n=k0.shape[0]
    return jax.pure_callback(_packed_callback,jax.ShapeDtypeStruct((2*n+1,),jnp.complex128),
                             k0,k1,k2,jax.lax.stop_gradient(reference),vmap_method='sequential')


@jax.custom_jvp
def simple_root(k0,k1,k2,reference):
    """A local simple eigenvalue selected near a fixed reference.

    This differentiates matrix coefficients, not branch sorting, eigenvector
    histories or an unresolved crossing. It rejects an ill-conditioned root.
    Physical matrix/background derivatives must be provided and validated
    separately. Second derivatives through this callback are unsupported.
    """
    return _pack(k0,k1,k2,reference)[0]


@simple_root.defjvp
def _root_jvp(primals,tangents):
    k0,k1,k2,reference=primals
    d0,d1,d2,_=tangents
    pack=_pack(k0,k1,k2,reference)
    n=k0.shape[0]
    sigma,x,y=pack[0],pack[1:n+1],pack[n+1:]
    numerator=jnp.vdot(y,(d0+sigma*d1+sigma*sigma*d2)@x)
    denominator=jnp.vdot(y,(k1+2*sigma*k2)@x)
    return sigma,-numerator/denominator
