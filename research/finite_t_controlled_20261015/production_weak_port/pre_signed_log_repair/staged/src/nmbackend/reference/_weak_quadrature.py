"""Positive/log deterministic quadrature for the free charged-current reference."""
from __future__ import annotations

import math
import numbers
import time
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

NAMES = ('number', 'neutrino_energy', 'charged_lepton_energy', 'proton_energy',
         'neutron_energy', 'material_energy_loss')


def finite_real(x, name='value'):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
        raise ValueError(name + ' must be a finite non-Boolean real')
    try:
        value = float(x)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + ' is not representable') from error
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return value


def log_fd(x):
    return -np.logaddexp(0., -np.asarray(x))


def kinetic_grid(order, temperature, upper):
    x, w = leggauss(order)
    y = (x + 1) * math.sqrt(upper / temperature) / 2
    return temperature*y*y, w*math.sqrt(upper/temperature)*temperature*y


@dataclass(frozen=True)
class PositiveQuantity:
    """Positive support with a finite log, or an explicitly absent occupation."""
    value: object
    log: object
    reason: str

    def __post_init__(self):
        if self.log is None:
            if type(self.value) is not float or self.value != 0. or self.reason != 'absent_initial_neutrino_occupation':
                raise ValueError('Only proven absent incoming radiation has exact zero')
            return
        z = finite_real(self.log, 'log')
        try:
            v = math.exp(z)
        except OverflowError:
            v = math.inf
        expected = 'underflow' if v == 0 else 'overflow' if not math.isfinite(v) else 'represented'
        if self.reason != expected:
            raise ValueError('Representation reason inconsistent with physical log')
        if expected == 'represented':
            if finite_real(self.value) != v:
                raise ValueError('Value and positive log disagree')
        elif self.value is not None:
            raise ValueError('Unrepresented ordinary value must be None')


@dataclass(frozen=True)
class SignedQuantity:
    value: object
    log_abs: object
    sign: int

    def __post_init__(self):
        if type(self.sign) is not int or self.sign not in (-1, 0, 1):
            raise ValueError('Literal signed log sign required')
        if self.sign == 0:
            if type(self.value) is not float or self.value != 0. or self.log_abs is not None:
                raise ValueError('Exact numerical cancellation needs zero sign and no log')
        else:
            p = positive(self.log_abs)
            if self.value is not None:
                finite_real(self.value, 'signed value')
            if self.value != (None if p.value is None else self.sign*p.value):
                raise ValueError('Signed ordinary representation inconsistent')


def positive(log_value=None):
    if log_value is None:
        return PositiveQuantity(0., None, 'absent_initial_neutrino_occupation')
    z = finite_real(log_value, 'positive log')
    try:
        v = math.exp(z)
    except OverflowError:
        v = math.inf
    reason = 'underflow' if v == 0 else 'overflow' if not math.isfinite(v) else 'represented'
    return PositiveQuantity(v if reason == 'represented' else None, z, reason)


def signed(log_abs, sign):
    sign = int(sign)
    if sign == 0:
        return SignedQuantity(0., None, 0)
    p = positive(float(log_abs))
    return SignedQuantity(None if p.value is None else sign*p.value, p.log, sign)


class Timer:
    def __init__(self, seconds):
        self.start = time.monotonic()
        self.seconds = seconds

    def check(self):
        if time.monotonic()-self.start > self.seconds:
            raise TimeoutError('Incomplete physical quadrature: wall budget exceeded; no rate returned')


class Accumulator:
    def __init__(self):
        self.forward = [[] for _ in NAMES]
        self.backward = [[] for _ in NAMES]
        self.nets = [[] for _ in NAMES]
        self.signs = [[] for _ in NAMES]
        self.entropy = []
        self.kms = 0.

    def add(self, base, energies, lf, lb, expected):
        if not np.all(np.isfinite(base)) or not np.all(np.isfinite(energies)) or np.any(energies <= 0):
            raise ArithmeticError('Unresolved positive physical event measure/energy')
        if not np.all(np.isfinite(lf)) or (lb is not None and not np.all(np.isfinite(lb))):
            raise ArithmeticError('Unresolved positive FD product')
        if lb is None:
            netlog = base+lf
            signs = np.ones_like(netlog)
        else:
            delta = lf-lb
            self.kms = max(self.kms, float(np.max(abs(delta-expected))))
            signs = np.sign(delta)
            mask = signs != 0
            netlog = np.full_like(base+lf, -np.inf)
            netlog[mask] = (base+np.maximum(lf, lb))[mask]+np.log(-np.expm1(-abs(delta[mask])))
            if np.any(mask):
                self.entropy.append(float(logsumexp(netlog[mask]+np.log(abs(delta[mask])))))
        for i in range(len(NAMES)):
            le = np.log(energies[:, i])
            self.forward[i].append(float(logsumexp(base+lf+le)))
            if lb is not None:
                self.backward[i].append(float(logsumexp(base+lb+le)))
            z, s = logsumexp(netlog+le, b=signs, return_sign=True)
            if s:
                self.nets[i].append(float(z))
                self.signs[i].append(float(s))

    def finish(self, radiation):
        if not self.forward[0] or not math.isfinite(float(logsumexp(self.forward[0]))):
            raise ArithmeticError('Finite-T positive integrated channel unresolved, not exact empty support')
        f = tuple(positive(float(logsumexp(v))) for v in self.forward)
        b = tuple(positive(float(logsumexp(v))) if v else positive() for v in self.backward)
        n = []
        for logs, signs in zip(self.nets, self.signs):
            z, s = logsumexp(logs, b=signs, return_sign=True) if logs else (0., 0)
            n.append(signed(z, s))
        entropy = (signed(float(logsumexp(self.entropy)), 1) if self.entropy else signed(0., 0)) if radiation.mode != 'empty' else None
        if radiation.mode != 'empty' and not self.backward[0]:
            raise ArithmeticError('Missing positive inverse branch for finite FD radiation')
        return f, b, tuple(n), entropy, self.kms
