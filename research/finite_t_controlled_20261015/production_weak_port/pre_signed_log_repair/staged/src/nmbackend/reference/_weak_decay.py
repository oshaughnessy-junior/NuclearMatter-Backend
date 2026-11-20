"""Exact moving-neutron three-body decay/inverse; no lab rest-endpoint cut."""
from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_genlaguerre

from . import _weak_current as C
from ._weak_quadrature import Accumulator, Timer, log_fd


def pair_events(s, cosine):
    """Future pair Q=electron+antinu; current qGamma=proton−neutron=−Q."""
    root = np.sqrt(s)
    momentum = np.sqrt(((C.MN-C.MP)**2-s)*((C.MN+C.MP)**2-s))/(2*C.MN)
    pair_energy = (C.MN*C.MN-C.MP*C.MP+s)/(2*C.MN)
    proton_energy = C.MN-pair_energy
    lepton_p = (s-C.ME*C.ME)/(2*root)
    lepton_e = (s+C.ME*C.ME)/(2*root)
    neutron = np.tile([C.MN, 0., 0., 0.], (len(s), 1))
    proton = np.column_stack([proton_energy, np.zeros(len(s)), np.zeros(len(s)), -momentum])
    electron = np.column_stack([lepton_e, lepton_p*np.sqrt(1-cosine*cosine), np.zeros(len(s)), lepton_p*cosine])
    anti = np.column_stack([lepton_p, -electron[:, 1:]])
    beta = np.column_stack([np.zeros(len(s)), np.zeros(len(s)), momentum/pair_energy])
    return neutron, proton, C.boost(electron, beta), C.boost(anti, beta), momentum, lepton_p


def electron_events(energy, cosine):
    pe = np.sqrt((energy-C.ME)*(energy+C.ME))
    endpoint = ((C.MN-C.MP)*(C.MN+C.MP)+C.ME*C.ME)/(2*C.MN)
    denominator = C.MN-energy+pe*cosine
    enu = C.MN*(endpoint-energy)/denominator
    neutron = np.tile([C.MN, 0., 0., 0.], (len(energy), 1))
    electron = np.column_stack([energy, np.zeros(len(energy)), np.zeros(len(energy)), pe])
    anti = np.column_stack([enu, enu*np.sqrt(1-cosine*cosine), np.zeros(len(energy)), enu*cosine])
    return neutron, neutron-electron-anti, electron, anti, pe, enu, denominator


def rest_events(order, coordinate, point=False):
    x, w = leggauss(order)
    if coordinate == 'pair':
        theta = (x+1)*math.pi/4
        width = (C.MN-C.MP)**2-C.ME**2
        s = C.ME*C.ME+width*np.sin(theta)**2
        weights = width*2*np.sin(theta)*np.cos(theta)*w*math.pi/4
        ss, cc = np.meshgrid(s, x, indexing='ij')
        phase_weight = (weights[:, None]*w[None, :]).ravel()
        pn, pp, el, nu, p, pe = pair_events(ss.ravel(), cc.ravel())
        phase = phase_weight*p*pe/(128*math.pi**3*C.MN**2*np.sqrt(ss.ravel())*C.HBAR)
    else:
        endpoint = ((C.MN-C.MP)*(C.MN+C.MP)+C.ME*C.ME)/(2*C.MN)
        top = math.sqrt(endpoint-C.ME)
        z = (x+1)*top/2
        energy = C.ME+z*z
        weights = w*top*z
        ee, cc = np.meshgrid(energy, x, indexing='ij')
        phase_weight = (weights[:, None]*w[None, :]).ravel()
        pn, pp, el, nu, pe, enu, denominator = electron_events(ee.ravel(), cc.ravel())
        phase = phase_weight*pe*enu/(64*math.pi**3*C.MN*denominator*C.HBAR)
    weight = phase*C.decay_amplitude(pn, pp, el, nu, point=point)
    if np.any(weight <= 0) or not np.all(np.isfinite(weight)):
        raise ArithmeticError('Rest decay measure unresolved')
    return dict(neutron=pn, proton=pp, lepton=el, neutrino=nu, weight=weight)


def orient(rest, polar_order, azimuth_order):
    cosine, weights = leggauss(polar_order)
    phi = 2*math.pi*np.arange(azimuth_order)/azimuth_order
    co, ph = np.meshgrid(cosine, phi, indexing='ij')
    co, ph = co.ravel(), ph.ravel()
    si = np.sqrt(1-co*co)
    orientation_weight = np.repeat(weights/(2*azimuth_order), azimuth_order)
    result = {}
    for name in ('neutron', 'proton', 'lepton', 'neutrino'):
        v = rest[name]
        xx = v[:, 1, None]*np.cos(ph)-v[:, 2, None]*np.sin(ph)
        yy = v[:, 1, None]*np.sin(ph)+v[:, 2, None]*np.cos(ph)
        zz = np.broadcast_to(v[:, 3, None], xx.shape)
        result[name] = np.stack([np.broadcast_to(v[:, 0, None], xx.shape), co*xx+si*zz, yy, -si*xx+co*zz], axis=-1).reshape((-1, 4))
    result['weight'] = (rest['weight'][:, None]*orientation_weight).ravel()
    return result


def integrate(bath, radiation, controls):
    timer = Timer(controls.budget_seconds)
    rest = rest_events(controls.rest_order, controls.decay_coordinate)
    timer.check()
    rot = orient(rest, controls.orient_order, controls.decay_azimuth_order)
    nodes = []
    if controls.parent_coordinate == 'laguerre':
        x, w = roots_genlaguerre(controls.parent_order, .5)
        for xx, ww in zip(x, w):
            en = C.MN+bath.T*xx
            measure = ww*bath.T*math.sqrt(bath.T*(2*C.MN+bath.T*xx))*en/(math.pi**2*C.HC**3)
            nodes.append((en, measure, float(xx)))
    else:
        center = max(0., bath.mu_n-C.MN)
        if radiation.mode != 'empty':
            center = max(center, bath.mu_p+bath.mu_e+radiation.mu_antinue-C.MN)
        upper = center+controls.tail*bath.T
        momentum = lambda kinetic: math.sqrt(kinetic*(kinetic+2*C.MN))
        cuts = [momentum(z*bath.T) for z in (0, 1, 3, 8, 16, 30) if z*bath.T < upper]+[momentum(upper)]
        x, w = leggauss(controls.parent_order)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            for xx, ww in zip(x, w):
                p = lo+(xx+1)*(hi-lo)/2
                nodes.append((math.hypot(C.MN, p), p*p*ww*(hi-lo)/2/(math.pi**2*C.HC**3), 0.))
    acc = Accumulator()
    density = proper_density = max_four = max_energy = 0.
    expected = 0. if radiation.mode == 'empty' else (bath.mu_n-bath.mu_p-bath.mu_e-radiation.mu_antinue)/bath.T
    for en, measure, correction in nodes:
        timer.check()
        momentum = math.sqrt((en-C.MN)*(en+C.MN))
        gamma = en/C.MN
        beta = np.tile([0., 0., momentum/en], (len(rot['weight']), 1))
        lab = {name: C.boost(rot[name], beta) for name in ('neutron', 'proton', 'lepton', 'neutrino')}
        fn = float(log_fd((bath.mu_n-en)/bath.T))
        hn = float(log_fd((en-bath.mu_n)/bath.T))
        density += measure*math.exp(correction+fn)
        proper_density += measure*math.exp(correction+fn)/gamma
        fp = log_fd((bath.mu_p-lab['proton'][:, 0])/bath.T)
        hp = log_fd((lab['proton'][:, 0]-bath.mu_p)/bath.T)
        fe = log_fd((bath.mu_e-lab['lepton'][:, 0])/bath.T)
        he = log_fd((lab['lepton'][:, 0]-bath.mu_e)/bath.T)
        lf, lb = fn+hp+he, None
        if radiation.mode != 'empty':
            lf = lf+log_fd((lab['neutrino'][:, 0]-radiation.mu_antinue)/bath.T)
            lb = hn+fp+fe+log_fd((radiation.mu_antinue-lab['neutrino'][:, 0])/bath.T)
        loss = lab['neutron'][:, 0]-lab['proton'][:, 0]-lab['lepton'][:, 0]
        values = np.column_stack([np.ones(len(loss)), lab['neutrino'][:, 0], lab['lepton'][:, 0], lab['proton'][:, 0], lab['neutron'][:, 0], loss])
        base = np.log(rot['weight'])+math.log(measure/gamma)+correction
        acc.add(base, values, lf, lb, expected)
        max_four = max(max_four, float(np.max(abs(lab['neutron']-lab['proton']-lab['lepton']-lab['neutrino']))))
        max_energy = max(max_energy, float(np.max(abs(loss-lab['neutrino'][:, 0]))))
    timer.check()
    return acc.finish(radiation), dict(parent_particle_density=density, parent_proper_density=proper_density,
                                      rest_rate=float(sum(rest['weight'])), rest_neutrino_energy=float(rest['weight']@rest['neutrino'][:, 0]),
                                      max_four_residual=max_four, max_event_energy_residual=max_energy,
                                      points=len(nodes)*len(rot['weight']))
