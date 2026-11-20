"""Research reference: finite-mass leptons, classical static free nucleons.

Not a dense-matter or finite-recoil Urca calculation. See TECHNICAL_NOTE.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit

HBAR = 6.582119569e-22  # MeV s
HBARC = 197.3269804  # MeV fm
GF = 1.1663785e-11  # MeV^-2; PDG 2025
VUD = 0.974  # declared benchmark, not a precision CKM recommendation
GA = 1.2723  # declared benchmark axial coupling
ME = 0.51099895069  # MeV, electron rest energy
Q = 1.29333236  # MeV, neutron minus proton rest energy
MEV_TO_ERG = 1.602176634e-6
FM3_TO_CM3_DENSITY = 1e39
RATE_PREFACTOR = GF**2 * VUD**2 * (1 + 3 * GA**2) / (2 * np.pi**3 * HBAR)
DOS_PREFACTOR = 1 / (2 * np.pi**2 * HBARC**3)  # one neutrino helicity


@dataclass(frozen=True)
class State:
    T: float  # k_B T in MeV
    mu_e: float  # TOTAL electron chemical potential including rest energy
    n_n: float  # fm^-3, classical target density
    n_p: float  # fm^-3

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.T, self.mu_e, self.n_n, self.n_p)):
            raise ValueError("State entries must be finite")
        if self.T <= 0 or min(self.n_n, self.n_p) < 0:
            raise ValueError("Require T>0 and nonnegative target densities")


def occupation(E, T, mu):
    return expit((mu - np.asarray(E)) / T)


def spectral(state: State, E_nu, channel: str):
    """Return (spontaneous j, true chi) in s^-1 for C=j(1-f)-chi*f.

    Target recoil, thermal velocities and nucleon Pauli blocking are omitted.
    Electron final-state blocking is retained for neutron beta decay/absorption.
    """
    E_nu = np.asarray(E_nu, dtype=float)
    if np.any(~np.isfinite(E_nu)) or np.any(E_nu < 0):
        raise ValueError("Neutrino energy must be finite and nonnegative")
    if channel == "electron_capture":
        Ee = E_nu + Q
        emit_density, abs_density = state.n_p, state.n_n
        f = occupation(Ee, state.T, state.mu_e)
        emit_block, abs_block = f, occupation(-Ee, state.T, -state.mu_e)
    elif channel == "positron_capture":
        Ee = E_nu - Q
        emit_density, abs_density = state.n_n, state.n_p
        f = occupation(Ee, state.T, -state.mu_e)
        emit_block, abs_block = f, occupation(-Ee, state.T, state.mu_e)
    elif channel == "neutron_decay":
        Ee = Q - E_nu
        emit_density, abs_density = state.n_n, state.n_p
        f = occupation(Ee, state.T, state.mu_e)
        emit_block, abs_block = occupation(-Ee, state.T, -state.mu_e), f
    else:
        raise ValueError(f"Unknown channel: {channel}")
    allowed = Ee >= ME
    pe = np.sqrt(np.maximum(Ee**2 - ME**2, 0))
    phase = np.where(allowed, Ee * pe, 0)
    scale = RATE_PREFACTOR / DOS_PREFACTOR * phase
    return scale * emit_density * emit_block, scale * abs_density * abs_block


@lru_cache(maxsize=16)
def _nodes(order):
    return leggauss(order)


def target_moments(T, mu_e, channel, order=96, tail=70, mass=ME, q=Q):
    """Return number s^-1/target and neutrino MeV s^-1/target.

    Integrate E_e=mass+x^2 to regularize the lepton threshold. tail is the
    energy-tail cutoff in units of T beyond the largest relevant energy.
    mass/q overrides exist only for analytic-limit qualification.
    """
    if (not all(np.isfinite(v) for v in (T, mu_e, mass, q, tail, order))
            or T <= 0 or mass < 0 or q < 0 or order < 4 or tail <= 0
            or int(order) != order):
        raise ValueError("Invalid integration controls")
    if channel == "electron_capture":
        lower, upper = max(mass, q), max(mass, q, mu_e) + tail * T
    elif channel == "positron_capture":
        lower, upper = mass, max(mass, -mu_e) + tail * T
    elif channel == "neutron_decay":
        lower, upper = mass, q
        if upper <= lower:
            return np.array([0., 0.])
    else:
        raise ValueError(f"Unknown channel: {channel}")
    nodes, weights = _nodes(order)
    # Resolve an arbitrarily narrow Fermi surface with explicit energy panels.
    # A single interval can falsely appear converged at large mu/T.
    fermi_energy = -mu_e if channel == "positron_capture" else mu_e
    edges = sorted(set([lower, upper] + [fermi_energy + k*T for k in (-12, -4, 0, 4, 12)
                                       if lower < fermi_energy + k*T < upper]))
    xs, ws = [], []
    for elo, ehi in zip(edges[:-1], edges[1:]):
        a, b = np.sqrt(elo-mass), np.sqrt(ehi-mass)
        xs.append(a + .5*(nodes+1)*(b-a))
        ws.append(.5*(b-a)*weights)
    x, weights = np.concatenate(xs), np.concatenate(ws)
    Ee = mass + x*x
    pe = np.sqrt(np.maximum(Ee*Ee - mass*mass, 0))
    if channel == "electron_capture":
        Enu, block = Ee - q, occupation(Ee, T, mu_e)
    elif channel == "positron_capture":
        Enu, block = Ee + q, occupation(Ee, T, -mu_e)
    else:
        Enu, block = q - Ee, occupation(-Ee, T, -mu_e)
    base = Ee * pe * Enu**2 * block * 2*x
    scale = RATE_PREFACTOR
    return scale * np.array([weights @ base, weights @ (base * Enu)])


def source_terms(state: State, order=96):
    """Transparent isotropic emission; no external neutrino occupation.

    Number in fm^-3 s^-1, energy in MeV fm^-3 s^-1; dYe/dt per baryon.
    Charge neutrality is a separate thermo solve, not imposed here.
    """
    ec = state.n_p * target_moments(state.T, state.mu_e, "electron_capture", order)
    pc = state.n_n * target_moments(state.T, state.mu_e, "positron_capture", order)
    nd = state.n_n * target_moments(state.T, state.mu_e, "neutron_decay", order)
    nb = state.n_n + state.n_p
    return {
        "electron_capture": ec.tolist(), "positron_capture": pc.tolist(),
        "neutron_decay": nd.tolist(),
        "dYe_dt_s-1": float((pc[0] + nd[0] - ec[0]) / nb) if nb else 0.,
        "nu_e_number_fm-3_s-1": float(ec[0]),
        "antinu_e_number_fm-3_s-1": float(pc[0] + nd[0]),
        "neutrino_loss_MeV_fm-3_s-1": float(ec[1] + pc[1] + nd[1]),
        "neutrino_loss_erg_cm-3_s-1": float((ec[1]+pc[1]+nd[1]) * MEV_TO_ERG * FM3_TO_CM3_DENSITY),
    }
