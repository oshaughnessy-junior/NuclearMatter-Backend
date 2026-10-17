"""Actual three-channel total-source assembly and instantaneous first law.

No spectral reconstruction, source cache, integration or complete weak inventory.
"""
from __future__ import annotations
from dataclasses import asdict, fields, is_dataclass
import math
from scipy.special import logsumexp
from . import free_matter as F
from . import weak_reference as W
from ._weak_quadrature import signed, finite_real

_TOKEN = object()  # internal same-process factory receipt; not a security sandbox


class UnrepresentedSourceError(ArithmeticError):
    """The exact log-bearing source is retained, but no finite RHS is available."""
    def __init__(self, source):
        self.source = source
        super().__init__('Source has unrepresented ordinary arithmetic; log record retained')


def _plain(value):
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)
                if not f.name.startswith('_')}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _payload(source):
    return _plain(source)


def _sum(terms):
    logs, signs = [], []
    for factor, q in terms:
        if q.sign:
            logs.append(q.log_abs)
            signs.append(factor*q.sign)
    if not logs:
        return signed(0., 0)
    z, sign = logsumexp(logs, b=signs, return_sign=True)
    return signed(float(z), int(sign)) if sign else signed(0., 0)


def _negative(q):
    return signed(q.log_abs, -q.sign) if q.sign else signed(0., 0)


def _gross_log(results, moment):
    return float(logsumexp([q.log for r in results for q in
                           (r.forward[moment], r.backward[moment]) if q.log is not None]))


def _residual(terms, scale):
    q = _sum(terms)
    return q.sign*math.exp(q.log_abs-scale) if q.sign else 0.


def _verify_result(result, channel, bath, radiation, control):
    if type(result) is not W.MomentResult or result.channel != channel:
        raise ValueError('Missing/wrong channel result; incomplete source rejected')
    for actual, expected in [(result.bath, bath), (result.radiation, radiation),
                              (result.controls, control), (result.current, W.Current())]:
        if type(actual) is not type(expected) or F._identity(asdict(actual)) != F._identity(asdict(expected)):
            raise ValueError('Channel bath/current/radiation/control identity mismatch')
    if any(type(v) is not tuple or len(v) != 6 for v in
           (result.forward, result.backward, result.net)):
        raise ValueError('Incomplete channel moment tuple')
    for direction, quantities in [('forward', result.forward), ('backward', result.backward)]:
        for q in quantities:
            if type(q) is not W.PositiveQuantity:
                raise ValueError('Expected physical positive/log quantity')
            W.PositiveQuantity(q.value, q.log, q.reason)
            absent = q.log is None
            if absent != (direction == 'backward' and radiation.mode == 'empty'):
                raise ArithmeticError('Only empty incoming radiation has absent inverse branch')
    for q in result.net:
        if type(q) is not W.SignedQuantity:
            raise ValueError('Expected signed/log net quantity')
        W.SignedQuantity(q.value, q.log_abs, q.sign)
    for i, net in enumerate(result.net):
        f, b = result.forward[i], result.backward[i]
        ff = signed(f.log, 1)
        bb = signed(b.log, 1) if b.log is not None else signed(0., 0)
        scale = float(logsumexp([q.log for q in (f, b) if q.log is not None]))
        if abs(_residual([(1, net), (-1, ff), (1, bb)], scale)) > 2e-8:
            raise ArithmeticError('Inconsistent positive branches and signed net')
    meta = dict(result.diagnostics)
    if (type(result.diagnostics) is not tuple or len(meta) != len(result.diagnostics)
            or type(meta.get('points')) is not int or meta.get('points', 0) <= 0
            or meta.get('parent_particle_density', 0) <= 0
            or not math.isfinite(finite_real(result.max_event_kms_error))
            or not 0 <= result.max_event_kms_error <= 1e-8):
        raise ArithmeticError('Incomplete/invalid physical quadrature diagnostics')
    for key, value in meta.items():
        if type(key) is not str or finite_real(value, key) < 0:
            raise ArithmeticError('Malformed/nonfinite quadrature diagnostics')
    expected = ((('neutron', 1), ('proton', -1), ('electron_net', -1), ('nue', 1))
                if channel == 'EC' else
                (('neutron', -1), ('proton', 1), ('positron', -1), ('electron_net', 1), ('antinue', 1))
                if channel == 'PC' else
                (('neutron', -1), ('proton', 1), ('electron_net', 1), ('antinue', 1)))
    if (type(result.particle_stoichiometry) is not tuple or
            any(type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str
                or type(pair[1]) is not int for pair in result.particle_stoichiometry)
            or result.particle_stoichiometry != expected):
        raise ValueError('Channel stoichiometry does not match declared reaction')
    for residual in (result.energy_ledger_scaled, result.direct_energy_ledger_scaled):
        if residual is not None and abs(finite_real(residual, 'ledger')) > 2e-8:
            raise ArithmeticError('Channel energy ledger rejected')
    if radiation.mode == 'FD':
        entropy = result.entropy_production
        if type(entropy) is not W.SignedQuantity or entropy.sign < 0:
            raise ArithmeticError('FD physical entropy record missing/negative')
        W.SignedQuantity(entropy.value, entropy.log_abs, entropy.sign)
    elif result.entropy_production is not None:
        raise ValueError('No total entropy derivative assigned to empty radiation')


def source_moments(state, *, radiation, controls):
    """Fresh full EC/PC/neutron-decay events at canonically checked matter state."""
    state = F._canonical(state)
    if type(radiation) is not W.Radiation or type(controls) is not F.SourceControls:
        raise ValueError('Explicit exact Radiation and SourceControls required')
    copy = W.Radiation(**asdict(radiation))
    if F._identity(asdict(copy)) != F._identity(asdict(radiation)):
        raise ValueError('Altered/noncanonical radiation record')
    F.SourceControls(controls.EC, controls.PC, controls.NEUTRON_DECAY)
    bath = W.FreeElectronBath(state.temperature_mev, state.neutron.chemical_potential,
                              state.proton.chemical_potential, state.electron.chemical_potential)
    if radiation.mode == 'FD' and radiation.temperature != bath.T:
        raise ValueError('Exact common radiation/material T required')
    results = []
    for channel in W.CHANNELS:
        control = getattr(controls, channel)
        result = W.channel_moments(channel, bath, radiation, control)
        _verify_result(result, channel, bath, radiation, control)
        results.append(result)
    results = tuple(results)
    ec, pc, decay = results
    r = _sum([(-1, ec.net[0]), (1, pc.net[0]), (1, decay.net[0])])
    n, electron = _negative(r), r
    nue, anti = ec.net[0], _sum([(1, pc.net[0]), (1, decay.net[0])])
    q = _sum([(1, result.net[1]) for result in results])
    loss = _sum([(1, result.net[5]) for result in results])
    material = _negative(q)  # included-closure energy source; direct loss checked below
    number_scale, energy_scale = _gross_log(results, 0), _gross_log(results, 1)
    # Literal material energy changes from original p/n/charged moments.
    terms = []
    for result in results:
        if result.channel == 'EC':
            terms.extend([(1, result.net[4]), (-1, result.net[3]), (-1, result.net[2])])
        elif result.channel == 'PC':
            terms.extend([(1, result.net[3]), (-1, result.net[4]), (-1, result.net[2])])
        else:
            terms.extend([(1, result.net[3]), (1, result.net[2]), (-1, result.net[4])])
    ledgers = F.SourceLedgers(_residual([(1, r), (1, n)], number_scale),
                              _residual([(1, r), (-1, electron)], number_scale),
                              _residual([(1, electron), (1, nue), (-1, anti)], number_scale),
                              _residual(terms+[(1, q)], energy_scale),
                              _residual([(1, loss), (-1, q)], energy_scale))
    if any(not math.isfinite(v) or abs(v) > ledgers.tolerance
           for v in (ledgers.baryon_scaled, ledgers.charge_scaled,
                     ledgers.electron_lepton_scaled, ledgers.material_energy_scaled,
                     ledgers.direct_energy_scaled)):
        raise ArithmeticError('Included three-channel energy/Le ledger failed')
    if F._provenance() != state.provenance:
        raise RuntimeError('Prerequisite/binder source changed during quadrature')
    source = F.FreeNPESource(state.identity, bath, radiation, controls, results,
                             r, n, electron, nue, anti, q, q, material, ledgers,
                             W.CHANNELS, state.provenance)
    object.__setattr__(source, '_receipt', (_TOKEN, source.identity))
    return source


def instantaneous_rhs(state, source, *, baryon_density_dot_fm3_s):
    """Fixed-mass first law with prescribed density derivative; no integrator."""
    state = F._canonical(state)
    if type(source) is not F.FreeNPESource or source.state_identity != state.identity:
        raise ValueError('Source belongs to a different/invalid canonical state')
    receipt = source._receipt
    if (type(receipt) is not tuple or len(receipt) != 2 or receipt[0] is not _TOKEN
            or receipt[1] != source.identity or source.provenance != state.provenance):
        raise ValueError('Source is stale/altered or lacks actual factory receipt')
    ndot = finite_real(baryon_density_dot_fm3_s, 'nBdot')
    if any(q.value is None for r in source.channel_results
           for qs in (r.forward, r.backward, r.net) for q in qs):
        raise UnrepresentedSourceError(source)
    R, Q = source.proton_number_source.value, source.neutrino_energy_escape.value
    if R is None or Q is None:
        raise UnrepresentedSourceError(source)
    ydot = R/state.baryon_density_fm3
    work = (state.energy_density+state.pressure)/state.baryon_density_fm3*ndot
    tdot = (work-state.denergy_dnB_fixed_T_Ye*ndot
            -state.denergy_dYe_fixed_nB_T*ydot-Q)/state.cv_fixed_nB_Ye
    affinity = state.proton.chemical_potential+state.electron.chemical_potential-state.neutron.chemical_potential
    entropy = (-Q-affinity*R)/(state.baryon_density_fm3*state.temperature_mev)
    if not all(math.isfinite(x) for x in (ydot, tdot, entropy, work)):
        raise ArithmeticError('Unrepresented instantaneous response')
    return F.MatterDerivative(tdot, ydot, ndot, entropy, work, state.identity, source.identity)
