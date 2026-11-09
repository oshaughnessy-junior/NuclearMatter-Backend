"""Mode-side exchange of conventional RNS geometry and perturbative closure.

This prepares inputs; it does not assemble an Einstein-fluid action or certify
background accuracy for a radiative eigenvalue calculation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from nmbackend.rotating.rns import RNSBackground
from .pencil import ModeError


@dataclass(frozen=True)
class ModeBackground:
    equilibrium: RNSBackground
    sound_speed_sq: np.ndarray
    closure: str
    response_source: str

    def __post_init__(self):
        if not isinstance(self.equilibrium,RNSBackground):
            raise ModeError('This adapter accepts conventional RNS backgrounds only')
        if self.closure not in ('barotropic','frozen-composition','finite-reaction') or not self.response_source:
            raise ModeError('Declare perturbative closure and response provenance')
        if self.closure=='finite-reaction':
            raise ModeError('Finite-reaction closure needs a frequency-dependent response, not scalar sound speed')
        fields=self.equilibrium.fields
        if not {'rho','gama','alpha','omega','energy','pressure','velocity_sq','enthalpy'}<=fields.keys():
            raise ModeError('Missing equilibrium fields needed by the mode interface')
        shape=fields['energy'].shape
        response=np.broadcast_to(np.asarray(self.sound_speed_sq,dtype=float),shape).copy()
        matter=fields['energy']>0
        if not np.any(matter) or np.any(~np.isfinite(response[matter])) or np.any(response[matter]<=0) or np.any(response[matter]>1):
            raise ModeError('Require 0 < adiabatic dp/de <= 1 throughout matter')
        response[~matter]=0.
        response.setflags(write=False)
        object.__setattr__(self,'sound_speed_sq',response)


def rns_polytropic_background(star):
    """Supply the exact barotropic polytrope derivative, not a composition closure."""
    if not isinstance(star,RNSBackground) or 'Gamma' not in star.metadata:
        raise ModeError('Require the polytropic RNS export with Gamma metadata')
    eps,p=star.fields['energy'],star.fields['pressure']
    response=star.metadata['Gamma']*p/np.where(eps+p>0,eps+p,1.)
    return ModeBackground(star,response,'barotropic','analytic relativistic polytrope dp/de=Gamma*p/(epsilon+p)')
