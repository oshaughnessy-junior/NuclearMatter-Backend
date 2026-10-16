"""Strict spectral coefficients and explicitly finite-range reconstruction.

One active helicity, no external radiation factor inside j or true chi.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp

from . import _weak_current as C
from ._weak_quadrature import (finite_real, positive, signed, log_fd, Timer,
                               PositiveQuantity, SignedQuantity)

FIXED_BATH = (2., 919.8287578100662, 916.8449303084382, .17446333200826397)
LIGHT_CM_S = 2.99792458e10


@dataclass(frozen=True)
class SpectralQuantity:
    """Finite positive log, or exactly empty ANALYTIC differential support."""
    value: object
    log: object
    support: str
    reason: str

    def __post_init__(self):
        if self.support == 'exact_kinematic_zero':
            if type(self.value) is not float or self.value != 0. or self.log is not None or self.reason != 'analytic_empty_particle_hole_support':
                raise ValueError('An exact differential zero requires the analytic empty-support witness')
        elif self.support == 'positive_support':
            finite_real(self.log, 'positive spectral log')
            p = positive(self.log)
            if self.value is not None:
                finite_real(self.value, 'spectral value')
            if self.value != p.value or self.reason != p.reason:
                raise ValueError('Spectral ordinary value/log/representation disagree')
        else:
            raise ValueError('Unresolved support is an exception, not a coefficient')


def coefficient(log):
    p = positive(finite_real(log, 'integrated positive spectral log'))
    return SpectralQuantity(p.value, p.log, 'positive_support', p.reason)


def empty_coefficient():
    return SpectralQuantity(0., None, 'exact_kinematic_zero', 'analytic_empty_particle_hole_support')


@dataclass(frozen=True)
class SpectralControls:
    """Selected deterministic rules; bounds are not an accuracy certificate."""
    lepton_order: int = 8
    angle_order: int = 12
    energy_order: int = 12
    phi_order: int = 4
    tail: float = 50.
    outer_tail: float = 30.
    parent_order: int = 8
    cone_order: int = 10
    electron_order: int = 10
    decay_phi_order: int = 6
    budget_seconds: float = 110.

    def __post_init__(self):
        for name in ('lepton_order', 'angle_order', 'energy_order', 'phi_order', 'parent_order', 'cone_order', 'electron_order', 'decay_phi_order'):
            if type(getattr(self, name)) is not int or not 4 <= getattr(self, name) <= 32:
                raise ValueError('Literal integer spectral orders4..32 required')
        for name in ('tail', 'outer_tail', 'budget_seconds'):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if not 30 <= self.tail <= 100 or not 20 <= self.outer_tail <= 60 or not 0 < self.budget_seconds <= 120:
            raise ValueError('Tail30..100,outertail20..60,wall budget(0,120] required')
        if self.cone_order*self.electron_order*self.decay_phi_order > 250000:
            raise ValueError('Bounded spectral event array exceeded')

    @classmethod
    def fine(cls):
        return cls(lepton_order=12, angle_order=16, energy_order=16, phi_order=6,
                   tail=70., outer_tail=40., parent_order=12, cone_order=16,
                   electron_order=16, decay_phi_order=10)


@dataclass(frozen=True)
class SpectralResult:
    channel: str
    bath: object
    energy: float
    j: SpectralQuantity
    true_chi: SpectralQuantity
    controls: SpectralControls
    geometry: tuple
    differential_coordinates: object = None
    current_source: str = C.SOURCE
    complete: bool = True
    units: str = 's^-1 per occupation'

    def __post_init__(self):
        validate(self.channel, self.bath, self.energy, self.controls)
        if type(self.complete) is not bool or not self.complete or self.current_source != C.SOURCE:
            raise ValueError('Incomplete/source-mismatched spectral result')
        if not isinstance(self.j, SpectralQuantity) or not isinstance(self.true_chi, SpectralQuantity):
            raise ValueError('Strict direct spectral branches required')
        if self.differential_coordinates is None:
            if self.j.support != 'positive_support' or self.true_chi.support != 'positive_support' or self.units != 's^-1 per occupation':
                raise ValueError('Finite-T integrated j AND true chi require positive finite logs')
        else:
            if self.channel == 'NEUTRON_DECAY' or self.units != 's^-1 MeV^-1 per cosine':
                raise ValueError('Only capture has this differential measure')
            el, c = self.differential_coordinates
            differential_inputs(el, c)
            if self.j.support != self.true_chi.support:
                raise ValueError('Common physical support required for direct branches')
            if self.j.support == 'exact_kinematic_zero':
                if self.geometry != ('analytic_empty_particle_hole_support',):
                    raise ValueError('Missing analytic empty-support witness')
                # A tag is not a support proof. Recheck the SAME declared
                # species/physical coordinates with the unchanged cheap shell
                # algebra; singular/degenerate coordinates remain unresolved.
                from ._weak_spectral_capture import support
                pe = math.sqrt((el-C.ME)*(el+C.ME))
                omega = self.energy-el
                q = math.sqrt((self.energy-pe)**2+2*self.energy*pe*(1-c))
                ma, mb = (C.MN,C.MP) if self.channel == 'EC' else (C.MP,C.MN)
                if support(ma,mb,omega,q):
                    raise ValueError('Claimed empty differential has nonempty analytic support')

    @property
    def flavor(self):
        return 'nue' if self.channel == 'EC' else 'antinue'

    @property
    def relaxation(self):
        if self.j.log is None:
            return empty_coefficient()
        return coefficient(float(np.logaddexp(self.j.log, self.true_chi.log)))

    @property
    def absorption_cm_inverse(self):
        if self.differential_coordinates is not None:
            raise ValueError('Length conversion requires integrated true absorption')
        return coefficient(self.true_chi.log-math.log(LIGHT_CM_S))

    @property
    def absorption_km_inverse(self):
        return coefficient(self.absorption_cm_inverse.log+math.log(1e5))

    @property
    def kms_log_residual(self):
        if self.j.log is None:
            return None
        mu = self.bath.equilibrium_mu_nue if self.channel == 'EC' else self.bath.equilibrium_mu_antinue
        return self.j.log-self.true_chi.log+(self.energy-mu)/self.bath.T


def validate(channel, bath, energy, controls):
    # Import lazily to avoid a public/common module initialization cycle.
    from .weak_reference import FreeElectronBath
    if type(channel) is not str or channel not in ('EC', 'PC', 'NEUTRON_DECAY'):
        raise ValueError('Named EC/PC/NEUTRON_DECAY channel required')
    if type(bath) is not FreeElectronBath or tuple(getattr(bath, k) for k in ('T', 'mu_n', 'mu_p', 'mu_e')) != FIXED_BATH:
        raise ValueError('Spectra require the EXACT declared fixed T2 chemical bath; no EOS solve')
    if type(controls) is not SpectralControls:
        raise ValueError('Explicit SpectralControls required')
    controls.__post_init__()
    energy = finite_real(energy, 'incident energy')
    if not 0 < energy <= (1.4 if channel == 'NEUTRON_DECAY' else 60.):
        raise ValueError('Selected spectral request bounds violated; no physical endpoint implied')
    return energy


def differential_inputs(el, cosine):
    el = finite_real(el, 'total charged lepton energy')
    cosine = finite_real(cosine, 'lepton cosine')
    if el <= C.ME or not -1 <= cosine <= 1:
        raise ValueError('El>ME and cosine in[-1,1] required')
    return el, cosine


def dos(energy):
    energy = finite_real(energy, 'positive energy')
    if energy <= 0:
        raise ValueError('E=0 DOS limit is not qualified')
    return energy**2/(2*math.pi**2*C.HC**3)


@dataclass(frozen=True)
class EnergyRule:
    """Explicit finite positive energy nodes/weights and declared range."""
    nodes: tuple
    weights: tuple
    lower: float
    upper: float

    def __post_init__(self):
        if type(self.nodes) is not tuple or type(self.weights) is not tuple or not 1 <= len(self.nodes) <= 64 or len(self.nodes) != len(self.weights):
            raise ValueError('Immutable bounded complete node/weight tuples required')
        lo, hi = finite_real(self.lower), finite_real(self.upper)
        if not 0 <= lo < hi or any(not lo < finite_real(x) < hi for x in self.nodes) or any(finite_real(w) <= 0 for w in self.weights):
            raise ValueError('Positive interior nodes/weights and ordered finite coverage required')
        if any(a >= b for a, b in zip(self.nodes[:-1], self.nodes[1:])):
            raise ValueError('Strict distinct energy nodes required')
        if abs(math.fsum(self.weights)/(hi-lo)-1) > 1e-12:
            raise ValueError('Weights must cover the declared finite range')

    @classmethod
    def gauss(cls, lower, upper, order):
        from numpy.polynomial.legendre import leggauss
        lower, upper = finite_real(lower), finite_real(upper)
        if type(order) is not int or not 2 <= order <= 64 or not 0 <= lower < upper:
            raise ValueError('Finite ordered coverage and literal order2..64 required')
        x, w = leggauss(order)
        return cls(tuple(float(v) for v in lower+(x+1)*(upper-lower)/2),
                   tuple(float(v) for v in w*(upper-lower)/2), lower, upper)


@dataclass(frozen=True)
class ReconstructionResult:
    channel: str
    bath: object
    radiation: object
    rule: EnergyRule
    controls: SpectralControls
    nodes: tuple
    forward: tuple
    backward: tuple
    net: tuple
    entropy_production: object
    complete_finite_range: bool = True
    whole_moment_qualified: bool = False
    scope: str = 'finite-range quadrature PILOT; omitted incident-energy tails and continuum convergence OPEN'
    units: tuple = ('fm^-3 s^-1', 'MeV fm^-3 s^-1')

    def __post_init__(self):
        from .weak_reference import Radiation
        if self.complete_finite_range is not True or self.whole_moment_qualified is not False:
            raise ValueError('Only a complete finite-range pilot may be returned; whole moments OPEN')
        if self.units != ('fm^-3 s^-1', 'MeV fm^-3 s^-1') or self.scope != 'finite-range quadrature PILOT; omitted incident-energy tails and continuum convergence OPEN':
            raise ValueError('Finite-range scope and units must remain explicit')
        if type(self.rule) is not EnergyRule or type(self.radiation) is not Radiation:
            raise ValueError('Strict energy rule and radiation required')
        self.rule.__post_init__(); self.radiation.__post_init__()
        validate(self.channel,self.bath,self.rule.nodes[0],self.controls)
        if self.radiation.mode == 'FD' and self.radiation.temperature != self.bath.T:
            raise ValueError('Exactly common-T radiation required')
        if self.rule.upper > (1.4 if self.channel == 'NEUTRON_DECAY' else 60.):
            raise ValueError('Coverage outside declared spectral request domain')
        if type(self.nodes) is not tuple or len(self.nodes) != len(self.rule.nodes):
            raise ValueError('Every declared actual spectral node must be present')
        for energy,node in zip(self.rule.nodes,self.nodes):
            if type(node) is not SpectralResult:
                raise ValueError('Actual spectral result required at every node')
            node.__post_init__()
            if node.channel != self.channel or node.bath != self.bath or node.controls != self.controls or node.energy != energy or node.differential_coordinates is not None:
                raise ValueError('Mismatched state/control/channel/energy node')
        for branch in (self.forward,self.backward,self.net):
            if type(branch) is not tuple or len(branch) != 2:
                raise ValueError('Both number and energy moments required')
        for branch in (self.forward,self.backward):
            for v in branch:
                if type(v) is not PositiveQuantity:
                    raise ValueError('Strict positive/log branch moments required')
                v.__post_init__()
        if any(v.log is None for v in self.forward):
            raise ValueError('Positive integrated spectral emission cannot be empty')
        if self.radiation.mode == 'FD':
            if any(v.log is None for v in self.backward) or self.entropy_production is None:
                raise ValueError('Positive FD inverse branch and entropy required')
        elif any(v.log is not None for v in self.backward) or self.entropy_production is not None:
            raise ValueError('Empty radiation inverse SOURCE zero and entropy undefined')
        for v in self.net:
            if type(v) is not SignedQuantity:
                raise ValueError('Strict directly accumulated signed moments required')
            v.__post_init__()
        if self.entropy_production is not None:
            if type(self.entropy_production) not in (PositiveQuantity,SignedQuantity):
                raise ValueError('Nonnegative direct FD entropy required')
            self.entropy_production.__post_init__()
            if isinstance(self.entropy_production,SignedQuantity) and self.entropy_production.sign != 0:
                raise ValueError('Signed FD entropy record is allowed only for exact numerical zero')
        # Canonical deterministic assembly from these SAME stored actual nodes,
        # rule and radiation: BOTH positive branches, nets and entropy matter.
        # This is internal coherence, not a proof of manually fabricated physics.
        expected = _assemble_spectral_nodes(self.channel,self.bath,self.radiation,self.rule,self.nodes)
        if (self.forward,self.backward,self.net,self.entropy_production) != expected:
            raise ValueError('Stored F/B R/Q, signed nets or entropy disagree with actual nodes/rule/radiation')


def reconstruct(channel, bath, radiation, rule, controls, evaluator):
    from .weak_reference import Radiation
    if type(rule) is not EnergyRule or type(radiation) is not Radiation:
        raise ValueError('Explicit immutable energy rule and radiation required')
    rule.__post_init__(); radiation.__post_init__()
    validate(channel, bath, rule.nodes[0], controls)
    if radiation.mode == 'FD' and radiation.temperature != bath.T:
        raise ValueError('Exactly common-T radiation required')
    if rule.upper > (1.4 if channel == 'NEUTRON_DECAY' else 60.):
        raise ValueError('Coverage exceeds the declared spectral request domain')
    timer = Timer(controls.budget_seconds)
    rows = []
    for energy, weight in zip(rule.nodes, rule.weights):
        timer.check()
        # The same shared timer is passed through every coefficient calculation.
        row = evaluator(channel, bath, energy, controls, timer)
        if not isinstance(row, SpectralResult) or row.differential_coordinates is not None or row.channel != channel or row.bath != bath or row.energy != energy or row.controls != controls:
            raise ArithmeticError('Incomplete or mismatched spectral node')
        row.__post_init__(); rows.append(row)
    timer.check()
    f,b,nets,ent = _assemble_spectral_nodes(channel,bath,radiation,rule,tuple(rows))
    return ReconstructionResult(channel,bath,radiation,rule,controls,tuple(rows),f,b,nets,ent)


def _assemble_spectral_nodes(channel,bath,radiation,rule,rows):
    """Pure canonical source arithmetic; no quadrature/current/EOS calls."""
    fs = [[], []]; bs = [[], []]; ns = [[], []]; signs = [[], []]; entropy = []
    for energy,weight,row in zip(rule.nodes,rule.weights,rows):
        base = math.log(weight)+2*math.log(energy)-math.log(2*math.pi**2*C.HC**3)
        if radiation.mode == 'empty':
            lf, lb = row.j.log, None
        else:
            mu = radiation.mu_nue if channel == 'EC' else radiation.mu_antinue
            lf = row.j.log+float(log_fd((energy-mu)/bath.T))
            lb = row.true_chi.log+float(log_fd((mu-energy)/bath.T))
        delta = None if lb is None else lf-lb
        sg = 1 if lb is None else int(np.sign(delta))
        netlog = lf if lb is None else (max(lf,lb)+math.log(-math.expm1(-abs(delta))) if sg else -math.inf)
        if lb is not None and sg:
            entropy.append(base+netlog+math.log(abs(delta)))
        for i in (0, 1):
            le = i*math.log(energy)
            fs[i].append(base+lf+le)
            if lb is not None: bs[i].append(base+lb+le)
            if sg: ns[i].append(base+netlog+le); signs[i].append(sg)
    f = tuple(positive(float(logsumexp(v))) for v in fs)
    b = tuple(positive(float(logsumexp(v))) if v else positive() for v in bs)
    nets = []
    for logs, ss in zip(ns, signs):
        z, s = logsumexp(logs, b=ss, return_sign=True) if logs else (0., 0)
        nets.append(signed(z, s))
    ent = None if radiation.mode == 'empty' else (positive(float(logsumexp(entropy))) if entropy else signed(0.,0))
    return f,b,tuple(nets),ent
