"""Exact moving-baryon EC/PC two-body events with actual FD blocking."""
from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss

from . import _weak_current as C
from ._weak_quadrature import Accumulator, Timer, kinetic_grid, log_fd


def electron_threshold(energy_p, momentum_p, cosine):
    """Incoming electron TOTAL energy at exact moving-proton threshold."""
    d = (C.MN*C.MN-C.MP*C.MP-C.ME*C.ME)/2
    a = energy_p*energy_p-momentum_p*momentum_p*cosine*cosine
    if energy_p*C.ME >= d or np.any(a <= 0) or np.any(d*d-a*C.ME*C.ME <= 0):
        raise ArithmeticError('Unresolved admitted electron threshold branch')
    return (energy_p*d+momentum_p*cosine*np.sqrt(d*d-a*C.ME*C.ME))/a


def integrate(channel, bath, radiation, controls):
    timer = Timer(controls.budget_seconds)
    ec = channel == 'EC'
    mass = C.MP if ec else C.MN
    final_mass = C.MN if ec else C.MP
    mu_initial = bath.mu_p if ec else bath.mu_n
    mu_final = bath.mu_n if ec else bath.mu_p
    mu_lepton = bath.mu_e if ec else -bath.mu_e
    mu_rad = radiation.mu_nue if ec else radiation.mu_antinue
    mu_rad = 0. if radiation.mode == 'empty' else mu_rad
    if ec:
        kmax = max(0., bath.mu_p-C.MP, bath.mu_n+max(0., mu_rad)-C.MP-C.ME)+controls.tail*bath.T
        emax = max(C.MN-C.MP, C.ME, bath.mu_e, bath.mu_n+max(0., mu_rad)-C.MP)+controls.tail*bath.T
    else:
        kmax = max(0., bath.mu_n-C.MN, bath.mu_p-C.MP)+controls.tail*bath.T+max(0., mu_rad)
        emax = C.ME+max(0., -bath.mu_e-C.ME, mu_rad-(C.MN-C.MP)-C.ME)+controls.tail*bath.T
    radial, radial_weights = kinetic_grid(controls.radial_order, bath.T, kmax)
    ci, wi = leggauss(controls.polar_order)
    co, wo = leggauss(controls.polar_order)
    phi = 2*math.pi*np.arange(controls.azimuth_order)/controls.azimuth_order
    xe, we = leggauss(controls.lepton_order)
    idx, jdx, outcos, outphi = np.meshgrid(np.arange(controls.lepton_order), np.arange(controls.polar_order), co, phi, indexing='ij')
    idx, jdx = idx.astype(int).ravel(), jdx.astype(int).ravel()
    outcos, outphi = outcos.ravel(), outphi.ravel()
    weights_angles = wi[jdx]*np.tile(np.repeat(wo, controls.azimuth_order), controls.lepton_order*controls.polar_order)*2*math.pi/controls.azimuth_order
    incos = ci[jdx]
    count = len(idx)
    acc = Accumulator()
    density = max_four = max_mass = max_energy = 0.
    expected = 0. if radiation.mode == 'empty' else (bath.mu_p+bath.mu_e-bath.mu_n-mu_rad)/bath.T if ec else (bath.mu_n-bath.mu_p-bath.mu_e-mu_rad)/bath.T
    for kin, wk in zip(radial, radial_weights):
        timer.check()
        energy_b = mass+kin
        momentum_b = math.sqrt(kin*(kin+2*mass))
        threshold = electron_threshold(energy_b, momentum_b, ci) if ec else np.full(len(ci), C.ME)
        if np.any(threshold >= emax):
            raise ArithmeticError('Numerical tail fails to enclose moving-lepton thresholds')
        ymax = np.sqrt((emax-threshold)/bath.T)
        y = (xe[:, None]+1)*ymax[None, :]/2
        electron = threshold[None, :]+bath.T*y*y
        jacobian = we[:, None]*ymax[None, :]*bath.T*y
        ee = electron[idx, jdx]
        pe = np.sqrt((ee-C.ME)*(ee+C.ME))
        weights = jacobian[idx, jdx]*weights_angles
        initial = np.tile([energy_b, 0., 0., momentum_b], (count, 1))
        lepton = np.column_stack([ee, pe*np.sqrt(1-incos*incos), np.zeros(count), pe*incos])
        total = initial+lepton
        s = C.dot(total, total)
        if np.any(s <= final_mass*final_mass):
            raise ArithmeticError('Threshold-resolved event is not physical')
        root = np.sqrt(s)
        beta = total[:, 1:]/total[:, 0, None]
        incoming_cm = C.boost(lepton, -beta)
        axis = incoming_cm[:, 1:]/np.linalg.norm(incoming_cm[:, 1:], axis=1)[:, None]
        ey = np.tile([0., 1., 0.], (count, 1))
        ex = np.cross(ey, axis)
        direction = outcos[:, None]*axis+np.sqrt(1-outcos*outcos)[:, None]*(np.cos(outphi)[:, None]*ex+np.sin(outphi)[:, None]*ey)
        pfinal = (s-final_mass*final_mass)/(2*root)
        neutrino = C.boost(np.column_stack([pfinal, pfinal[:, None]*direction]), beta)
        final = C.boost(np.column_stack([(s+final_mass*final_mass)/(2*root), -pfinal[:, None]*direction]), beta)
        pn, pp = (final, initial) if ec else (initial, final)
        amplitude = C.scalar_capture(channel, neutrino, pn, lepton, pp)
        # Spin2 baryon radial and spin2 lepton radial/relative angle, avg4 amplitude.
        # Moller flux and initial CM momentum cancel; no extra neutrino DOS.
        weight = (momentum_b*energy_b*wk/math.pi**2 * pe*ee/(2*math.pi**2)*weights
                  * amplitude*pfinal/(64*math.pi**2*root*energy_b*ee*C.HBAR*C.HC**3))
        if not np.all(np.isfinite(weight)) or np.any(weight <= 0):
            raise ArithmeticError('Positive physical event weights unresolved')
        fi = float(log_fd((mu_initial-energy_b)/bath.T))
        hi = float(log_fd((energy_b-mu_initial)/bath.T))
        fl = log_fd((mu_lepton-ee)/bath.T)
        hl = log_fd((ee-mu_lepton)/bath.T)
        ff = log_fd((mu_final-final[:, 0])/bath.T)
        hf = log_fd((final[:, 0]-mu_final)/bath.T)
        lf, lb = fi+fl+hf, None
        if radiation.mode != 'empty':
            lf = lf+log_fd((neutrino[:, 0]-mu_rad)/bath.T)
            lb = hi+hl+ff+log_fd((mu_rad-neutrino[:, 0])/bath.T)
        loss = energy_b+ee-final[:, 0]
        energies = np.column_stack([np.ones(count), neutrino[:, 0], ee, pp[:, 0], pn[:, 0], loss])
        acc.add(np.log(weight), energies, lf, lb, expected)
        density += momentum_b*energy_b*wk/(math.pi**2*C.HC**3)*math.exp(fi)
        max_four = max(max_four, float(np.max(abs(initial+lepton-final-neutrino))))
        max_mass = max(max_mass, float(np.max(abs(C.dot(final, final)-final_mass**2)))/final_mass**2)
        max_energy = max(max_energy, float(np.max(abs(loss-neutrino[:, 0]))))
    timer.check()
    return acc.finish(radiation), dict(parent_particle_density=density, max_four_residual=max_four,
                                      max_mass_relative=max_mass, max_event_energy_residual=max_energy,
                                      points=count*len(radial), incoming_kinetic_upper=kmax, lepton_total_upper=emax)
