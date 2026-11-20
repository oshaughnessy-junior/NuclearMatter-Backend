"""Detailed CPU total moments for three free electron charged-current channels.

Explicit chemical baths, physical masses and full Guo current. No EOS, spectral
opacity, hydro, JAX/AD or correlated/interacting Urca qualification is implied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from . import _weak_current as _C
from ._weak_quadrature import PositiveQuantity, SignedQuantity, finite_real

__all__ = ['FreeElectronBath', 'Current', 'Radiation', 'Controls', 'MomentResult',
           'PositiveQuantity', 'SignedQuantity', 'channel_moments', 'spectral_coefficients']
CHANNELS = ('EC', 'PC', 'NEUTRON_DECAY')


@dataclass(frozen=True)
class Current:
    """Fixed physical free current identity; no mutable model knobs."""
    source: str = field(default=_C.SOURCE, init=False)
    neutron_mass: float = field(default=_C.MN, init=False)
    proton_mass: float = field(default=_C.MP, init=False)
    electron_mass: float = field(default=_C.ME, init=False)
    weak_mass: float = field(default=_C.MW, init=False)


@dataclass(frozen=True)
class FreeElectronBath:
    """T/total mu in MeV; particle baryons and opposite e± mu, no neutrality solve."""
    T: float
    mu_n: float
    mu_p: float
    mu_e: float

    def __post_init__(self):
        for name in ('T', 'mu_n', 'mu_p', 'mu_e'):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if not 1 <= self.T <= 3:
            raise ValueError('Selected request domain T1..3 MeV only; no uniform accuracy certificate')
        if not all(-100 <= eta <= -5 for eta in ((self.mu_n-_C.MN)/self.T, (self.mu_p-_C.MP)/self.T)):
            raise ValueError('Particle baryon eta in [-100,-5] required')
        if abs(self.mu_e/self.T) > 3:
            raise ValueError('Selected pair-rich |total mu_e/T|<=3 required')

    @property
    def equilibrium_mu_nue(self):
        return self.mu_p+self.mu_e-self.mu_n

    @property
    def equilibrium_mu_antinue(self):
        return -self.equilibrium_mu_nue


@dataclass(frozen=True)
class Radiation:
    """Exactly empty, or common-T FD with independent nue/antinue chemical mu."""
    mode: str = 'empty'
    temperature: object = None
    mu_nue: object = None
    mu_antinue: object = None

    def __post_init__(self):
        if self.mode not in ('empty', 'FD') or type(self.mode) is not str:
            raise ValueError('Named empty or FD radiation required')
        if self.mode == 'empty':
            if any(v is not None for v in (self.temperature, self.mu_nue, self.mu_antinue)):
                raise ValueError('Exactly empty radiation has no temperature or chemical potentials')
        else:
            for name in ('temperature', 'mu_nue', 'mu_antinue'):
                object.__setattr__(self, name, finite_real(getattr(self, name), name))
            if self.temperature <= 0 or max(abs(self.mu_nue/self.temperature), abs(self.mu_antinue/self.temperature)) > 10:
                raise ValueError('Positive common T and independent bounded |mu_rad/T|<=10 required')

    @classmethod
    def empty(cls):
        return cls()

    @classmethod
    def fd(cls, temperature, mu_nue, mu_antinue):
        return cls('FD', temperature, mu_nue, mu_antinue)


@dataclass(frozen=True)
class Controls:
    """Declared quadrature controls; request bounds do not certify convergence."""
    radial_order: int = 20
    lepton_order: int = 32
    polar_order: int = 8
    azimuth_order: int = 8
    rest_order: int = 16
    parent_order: int = 20
    orient_order: int = 8
    decay_azimuth_order: int = 12
    decay_coordinate: str = 'pair'
    parent_coordinate: str = 'laguerre'
    tail: float = 60.
    budget_seconds: float = 110.

    def __post_init__(self):
        for name in ('radial_order', 'lepton_order', 'polar_order', 'azimuth_order',
                     'rest_order', 'parent_order', 'orient_order', 'decay_azimuth_order'):
            value = getattr(self, name)
            if type(value) is not int or not 4 <= value <= 64:
                raise ValueError('Literal integer quadrature orders4..64 required')
        for name in ('tail', 'budget_seconds'):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if not 30 <= self.tail <= 100 or not 0 < self.budget_seconds <= 120:
            raise ValueError('Tail30..100 and wall budget(0,120] required')
        if self.decay_coordinate not in ('pair', 'electron') or self.parent_coordinate not in ('laguerre', 'direct_p'):
            raise ValueError('Explicit named rest/parent coordinates required')
        if max(self.lepton_order*self.polar_order**2*self.azimuth_order,
               self.rest_order**2*self.orient_order*self.decay_azimuth_order) > 250000:
            raise ValueError('Bounded CPU event array limit250000 exceeded')

    @classmethod
    def base(cls, channel):
        if channel == 'EC':
            return cls(tail=50.)
        if channel == 'PC':
            return cls(radial_order=28, lepton_order=44, polar_order=10, azimuth_order=10)
        if channel == 'NEUTRON_DECAY':
            return cls()
        raise ValueError('Named EC/PC/NEUTRON_DECAY channel required')

    @classmethod
    def fine(cls, channel):
        if channel == 'EC':
            return cls(radial_order=28, lepton_order=48, polar_order=12, azimuth_order=12)
        if channel == 'PC':
            return cls(radial_order=32, lepton_order=52, polar_order=12, azimuth_order=12)
        if channel == 'NEUTRON_DECAY':
            return cls(rest_order=24, parent_order=28, orient_order=10, decay_azimuth_order=16,
                       decay_coordinate='electron', parent_coordinate='direct_p', tail=80.)
        raise ValueError('Named EC/PC/NEUTRON_DECAY channel required')


@dataclass(frozen=True)
class MomentResult:
    """Tuple moments: number,nu energy,charged energy,p energy,n energy,material loss.

    Volume number fm^-3s^-1; energy MeV fm^-3s^-1; entropy kB fm^-3s^-1.
    Positive branches retain finite logs when ordinary representation fails.
    """
    channel: str
    bath: FreeElectronBath
    radiation: Radiation
    controls: Controls
    current: Current
    forward: tuple
    backward: tuple
    net: tuple
    entropy_production: object
    max_event_kms_error: float
    diagnostics: tuple
    particle_stoichiometry: tuple
    energy_ledger_scaled: object
    direct_energy_ledger_scaled: object

    @property
    def names(self):
        return ('number', 'neutrino_energy', 'charged_lepton_energy', 'proton_energy',
                'neutron_energy', 'material_energy_loss')

    @property
    def units(self):
        return 'number fm^-3 s^-1; energy MeV fm^-3 s^-1; entropy kB fm^-3 s^-1'


def channel_moments(channel, bath, radiation=None, controls=None):
    """Actual full-current F/B totals; no interpolation, fit, EOS solve or cache."""
    if type(channel) is not str or channel not in CHANNELS:
        raise ValueError('Only named EC,PC,NEUTRON_DECAY total channels are available')
    if type(bath) is not FreeElectronBath:
        raise ValueError('Explicit FreeElectronBath required')
    radiation = Radiation.empty() if radiation is None else radiation
    controls = Controls.base(channel) if controls is None else controls
    if type(radiation) is not Radiation or type(controls) is not Controls:
        raise ValueError('Immutable explicit Radiation/Controls required')
    if radiation.mode == 'FD' and radiation.temperature != bath.T:
        raise ValueError('Only exact common-T FD radiation is admitted')
    if channel == 'NEUTRON_DECAY':
        from ._weak_decay import integrate
        raw, metadata = integrate(bath, radiation, controls)
    else:
        from ._weak_two_body import integrate
        raw, metadata = integrate(channel, bath, radiation, controls)
    forward, backward, net, entropy, kms = raw
    values = [q.value for q in net]
    scale = None if forward[1].value is None or backward[1].value is None else forward[1].value+backward[1].value
    energy = direct = None
    if scale is not None and scale > 0 and all(v is not None for v in values):
        if channel == 'EC':
            material = values[4]-values[3]-values[2]
        elif channel == 'PC':
            material = values[3]-values[4]-values[2]
        else:
            material = values[3]+values[2]-values[4]
        energy = (material+values[1])/scale
        direct = (values[1]-values[5])/scale
    stoichiometry = (('neutron', 1), ('proton', -1), ('electron_net', -1), ('nue', 1)) if channel == 'EC' else (
        (('neutron', -1), ('proton', 1), ('positron', -1), ('electron_net', 1), ('antinue', 1)) if channel == 'PC' else
        (('neutron', -1), ('proton', 1), ('electron_net', 1), ('antinue', 1)))
    return MomentResult(channel, bath, radiation, controls, Current(), forward, backward, net,
                        entropy, kms, tuple(sorted(metadata.items())), stoichiometry, energy, direct)


def spectral_coefficients(*args, **kwargs):
    """Not implemented in Stage A: no spectral/opacity result can be inferred."""
    raise NotImplementedError('Stage A provides TOTAL named-channel moments, no spectral j/chi or opacity')
