"""Fixed-mass free npe/pair matter for a restricted charged-current reference.

Instantaneous CPU reference only: no muons, interactions, paired EOS, AD or ODE.
Temperature has no rest offset; total energies and chemical potentials retain it.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Tuple, Optional

from scipy.special import kve
from . import finite_t_baryons as B
from . import finite_t_leptons as L
from . import weak_reference as W
from ._weak_quadrature import finite_real

MN, MP, ME, HC = 939.56542052, 938.27208816, .51099895069, 197.3269804
MODEL = 'free-npe-pair-physical-masses-PR51-52-54-no-muons-v1'
_PINS = {
    'finite_t_leptons': 'baeef78b174ecc0bcc1ab6016c839a8e8dc6a2ba915541d7fd994d994f01e719',
    'finite_t_baryons': 'f4c007f400df346811d4ce759ee8d96d6b74170c7ef1d0c4f5bc063f8018f65b',
    'weak_reference': 'a37508df389901e0f4e57ff028b850a33eb1beccfef79dcb67ce6b0b5f483172',
    '_weak_current': '2587e7c94bf6ceb86b04e55990faff593e1262444316ed77a920fb630bd7b5bc',
    '_weak_quadrature': '2d9a016f8ab629209856c40ac2ba4dac132280be1cfba82883586dd66decfd69',
    '_weak_two_body': '8d5c031ebb50de2f84fe6f1942ff70802f12f6bc1af53501729c6010f7401366',
    '_weak_decay': '21498bc563c7a4a7ae1d28b87410b3d5f0910bb54549ce86fd0ca20ab1657edc',
}


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _provenance():
    """Runtime file identity, not a sandbox or attestation of arbitrary Python memory."""
    rows = []
    constants = importlib.import_module('nmbackend.constants')
    cp = Path(constants.__file__).resolve()
    ch = hashlib.sha256(cp.read_bytes()).hexdigest()
    if ch != 'e4597ebdf07a7700f7e9b7282fca9bf78add2f1343c4c653199e09e916197317' or (B.HBARC, L.HBARC) != (HC, HC):
        raise RuntimeError('Prerequisite constants or HBARC changed')
    rows.append((constants.__name__, str(cp), ch))
    for name, expected in _PINS.items():
        module = importlib.import_module(__package__+'.'+name)
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError('Prerequisite source changed: '+name)
        rows.append((module.__name__, str(path), digest))
    for name in ('free_matter.py', '_free_matter_sources.py'):
        path = Path(__file__).resolve().with_name(name)
        rows.append((name, str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
    current = W.Current()
    if (current.neutron_mass, current.proton_mass, current.electron_mass,
            current.weak_mass) != (MN, MP, ME, (MN+MP)/2):
        raise RuntimeError('Fixed current/matter mass convention changed')
    return tuple(rows)


@dataclass(frozen=True)
class EOSControls:
    rtol: float = 2e-12
    tail: float = 60.

    def __post_init__(self):
        for name in ('rtol', 'tail'):
            object.__setattr__(self, name, finite_real(getattr(self, name), name))
        if not 0 < self.rtol < 1e-4 or not 30 <= self.tail <= 100:
            raise ValueError('EOS requires0<rtol<1e-4 and tail30..100')


@dataclass(frozen=True)
class AntiBound:
    species: str
    log_density_upper_bound: float
    log_fraction_upper_bound: float
    fraction_limit: float = 1e-8
    meaning: str = 'Omitted equilibrium FD antiparticle density <= spin2 Maxwell-Boltzmann bound; not added to EOS'


@dataclass(frozen=True)
class FreeNPEState:
    temperature_mev: float
    baryon_density_fm3: float
    Ye: float
    photons: str
    neutron: B.ParticleState
    proton: B.ParticleState
    electron: L.LeptonState
    muons_absent: bool
    model: str
    eos_controls: EOSControls
    energy_density: float
    pressure: float
    free_energy_density: float
    entropy_density: float
    cv_fixed_nB_Ye: float
    denergy_dYe_fixed_nB_T: float
    denergy_dnB_fixed_T_Ye: float
    h_n: float
    h_p: float
    h_e: float
    photon_energy_density: float
    photon_pressure: float
    photon_entropy_density: float
    photon_cv: float
    anti_bounds: Tuple[AntiBound, AntiBound]
    provenance: Tuple[Tuple[str, str, str], ...]

    @property
    def identity(self):
        return _identity(asdict(self))


@dataclass(frozen=True)
class SourceControls:
    EC: W.Controls
    PC: W.Controls
    NEUTRON_DECAY: W.Controls

    def __post_init__(self):
        for name in W.CHANNELS:
            control = getattr(self, name)
            if type(control) is not W.Controls:
                raise ValueError('Exact named-channel W.Controls required')
            copy = W.Controls(**asdict(control))
            if _identity(asdict(copy)) != _identity(asdict(control)):
                raise ValueError('Altered/noncanonical channel controls')

    @classmethod
    def fine(cls):
        return cls(*(W.Controls.fine(channel) for channel in W.CHANNELS))


@dataclass(frozen=True)
class SourceLedgers:
    baryon_scaled: float
    charge_scaled: float
    electron_lepton_scaled: float
    material_energy_scaled: float
    direct_energy_scaled: float
    tolerance: float = 2e-8


@dataclass(frozen=True)
class FreeNPESource:
    state_identity: str
    bath: W.FreeElectronBath
    radiation: W.Radiation
    controls: SourceControls
    channel_results: Tuple[W.MomentResult, W.MomentResult, W.MomentResult]
    proton_number_source: W.SignedQuantity
    neutron_number_source: W.SignedQuantity
    electron_net_number_source: W.SignedQuantity
    nue_number_source: W.SignedQuantity
    antinue_number_source: W.SignedQuantity
    neutrino_energy_escape: W.SignedQuantity
    material_energy_loss: W.SignedQuantity
    material_energy_source: W.SignedQuantity
    ledgers: SourceLedgers
    complete_channels: Tuple[str, str, str]
    provenance: Tuple[Tuple[str, str, str], ...]
    _receipt: object = field(default=None, init=False, repr=False, compare=False)

    @property
    def identity(self):
        from ._free_matter_sources import _payload
        return _identity(_payload(self))


@dataclass(frozen=True)
class MatterDerivative:
    temperature_dot_mev_s: float
    Ye_dot_s: float
    baryon_density_dot_fm3_s: float
    material_entropy_per_baryon_dot_s: float
    energy_work_density_source: float
    state_identity: str
    source_identity: str


def _anti(species, state):
    T, m, mu = state.temperature, state.mass, state.chemical_potential
    scaled = float(kve(2, m/T))
    if not math.isfinite(scaled) or scaled <= 0:
        raise ArithmeticError('Unresolved scaled-K2 anti-nucleon bound')
    log_bound = (2*math.log(m)+math.log(T)+math.log(scaled)-m/T-mu/T
                 -2*math.log(math.pi)-3*math.log(HC))
    ratio = log_bound-math.log(state.particle_density)
    if not all(math.isfinite(x) for x in (log_bound, ratio)) or ratio > math.log(1e-8):
        raise ValueError('Omitted '+species+' anti-nucleon bound exceeds1e-8')
    return AntiBound(species, log_bound, ratio)


def state_density(temperature_mev, baryon_density_fm3, Ye, *, photons,
                  controls=None):
    """Fresh particle-n/p and NET-electron roots; selected T1..3MeV slice.

    nB4e-8..2e-6, .05<Ye<.9; actual solved weak bath must also be admitted.
    Photon closure must be named absent or trapped_LTE. No beta equilibrium.
    """
    T, nB, y = [finite_real(v, n) for v, n in
                 [(temperature_mev, 'T'), (baryon_density_fm3, 'nB'), (Ye, 'Ye')]]
    if not 1 <= T <= 3 or not 4e-8 <= nB <= 2e-6 or not .05 < y < .9:
        raise ValueError('Selected T/nB/Ye rectangle exceeded; no extrapolation')
    if type(photons) is not str or photons not in ('absent', 'trapped_LTE'):
        raise ValueError('Explicit absent or trapped_LTE photons required')
    controls = EOSControls() if controls is None else controls
    if type(controls) is not EOSControls:
        raise ValueError('Exact immutable EOSControls required')
    # Reconstruct validation even if an immutable record was manually altered.
    copy = EOSControls(controls.rtol, controls.tail)
    if _identity(asdict(copy)) != _identity(asdict(controls)):
        raise ValueError('Altered/noncanonical EOS controls')
    prov = _provenance()
    kw = dict(rtol=controls.rtol, tail=controls.tail)
    neutron = B.state_density(T, (1-y)*nB, MN, **kw)
    proton = B.state_density(T, y*nB, MP, **kw)
    electron = L.state_density(T, y*nB, ME, **kw)
    for actual, target in [(neutron.particle_density, (1-y)*nB),
                           (proton.particle_density, y*nB), (electron.net_density, y*nB)]:
        if abs(actual/target-1) > 1e-9:
            raise ArithmeticError('Actual number root failed binder gate')
    if (neutron.mass, proton.mass, electron.mass) != (MN, MP, ME) or any(
            s.temperature != T for s in (neutron, proton, electron)):
        raise ArithmeticError('Component mass/temperature mismatch')
    for s in (neutron, proton, electron):
        if not all(math.isfinite(v) for v in asdict(s).values()):
            raise ArithmeticError('Unrepresented component state')
    W.FreeElectronBath(T, neutron.chemical_potential, proton.chemical_potential,
                       electron.chemical_potential)
    bounds = (_anti('neutron', neutron), _anti('proton', proton))
    eg = math.pi**2*T**4/(15*HC**3) if photons == 'trapped_LTE' else 0.
    pg, sg, cg = eg/3, 4*eg/(3*T), 4*eg/T
    e = math.fsum((neutron.energy_density, proton.energy_density, electron.energy_density, eg))
    p = math.fsum((neutron.pressure, proton.pressure, electron.pressure, pg))
    s = math.fsum((neutron.entropy_density, proton.entropy_density, electron.entropy_density, sg))
    cv = math.fsum((neutron.cv_fixed_particle_density, proton.cv_fixed_particle_density,
                    electron.cv_fixed_net_density, cg))
    hn = neutron.chemical_potential-T*neutron.dmu_dtemperature_fixed_particle_density
    hp = proton.chemical_potential-T*proton.dmu_dtemperature_fixed_particle_density
    he = electron.chemical_potential-T*electron.dmu_dtemperature_fixed_net_density
    ey = nB*(hp-hn+he)
    en = (1-y)*hn+y*(hp+he)
    if cv <= 0 or not all(math.isfinite(v) for v in (e, p, s, cv, hn, hp, he, ey, en)):
        raise ArithmeticError('Unresolved total thermodynamic response')
    return FreeNPEState(T, nB, y, photons, neutron, proton, electron, True, MODEL,
                        controls, e, p, e-T*s, s, cv, ey, en, hn, hp, he,
                        eg, pg, sg, cg, bounds, prov)


def _canonical(state):
    if type(state) is not FreeNPEState:
        raise ValueError('Factory FreeNPEState required')
    actual = state_density(state.temperature_mev, state.baryon_density_fm3, state.Ye,
                           photons=state.photons, controls=state.eos_controls)
    if state.identity != actual.identity:
        raise ValueError('Stale/forged state or component/model/control/source identity')
    return actual


def source_moments(state, *, radiation, controls):
    from ._free_matter_sources import source_moments as function
    return function(state, radiation=radiation, controls=controls)


def instantaneous_rhs(state, source, *, baryon_density_dot_fm3_s):
    from ._free_matter_sources import instantaneous_rhs as function
    return function(state, source, baryon_density_dot_fm3_s=baryon_density_dot_fm3_s)
