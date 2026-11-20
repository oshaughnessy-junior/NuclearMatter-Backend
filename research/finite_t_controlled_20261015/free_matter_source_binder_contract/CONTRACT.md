# Portable free-matter / charged-current source binder: next-slice contract

Design only, 16 October 2026. No binder implementation or new physical integration.
Future implementation requires explicit authorization and DIFFERENT numerical and
packaging review. Preserve all existing cold/cooling APIs and qualifications.

## 1. Exact prerequisites and verified source boundary

Use the ACTUAL explicitly imported `nmbackend.reference.finite_t_leptons` (draft
PR51, head11c6d41db14b906168efd2e018aca46d2e8d4d3c), `finite_t_baryons` (PR52,
head3125f3c450ddb1759132cf8367b4a649f15a75b3), and `weak_reference` (PR54,
headc17472b8a84655380ab64973ef883a6cd5fb2346). Fresh GitHub immutable-commit reads
of all three public module files match every staged byte; full responses and
SHA256s are retained in PUBLISHED_SOURCE_READS.json/API_INVENTORY.json. All three
drafts remain unmerged. A future stacked implementation must name a real base
containing their exact reviewed versions, or explicitly stage their dependency
integration separately. None is currently assumed present on main. No campaign
loaders, generic `study` aliases, stub packages or implicit unmerged dependencies.

PR55 spectra are NOT a dependency or part of this binder. Their selected T2
admission is narrower, soft PC support remains unresolved and beta local tail
controls fail. A complete moment adapter must not silently reconstruct totals
from those finite bands. It uses PR54 actual total event quadrature instead.

Published identities establish source equality, not accepted runtime coverage.
Minimum-supported versions, normal full CI, clean installation resolution, and
runtime import isolation must be closed separately for the assembled stack.

## 2. Existing callable and result contracts

Exact extracted public signatures, dataclass fields/defaults and methods are in
API_INVENTORY.json, generated statically without imports. Relevant functions are:

```
L.state_mu(temperature_mev, chemical_potential_mev, mass_mev=M_E,
           *, rtol=2e-12, tail=60.0) -> L.LeptonState
L.state_density(temperature_mev, net_density_fm3, mass_mev=M_E,
                *, rtol=2e-12, tail=60.0) -> L.LeptonState
L.thermal_energy_density(state, *, rtol=2e-12, tail=60.0) -> float
B.state_mu(temperature_mev, chemical_potential_mev, mass_mev=M_N,
           *, rtol=2e-12, tail=60.0) -> B.ParticleState
B.state_density(temperature_mev, particle_density_fm3, mass_mev=M_N,
                *, rtol=2e-12, tail=60.0) -> B.ParticleState
W.FreeElectronBath(T, mu_n, mu_p, mu_e)
W.Radiation.empty()
W.Radiation.fd(temperature, mu_nue, mu_antinue)
W.Controls.base(channel), W.Controls.fine(channel)
W.channel_moments(channel, bath, radiation=None, controls=None) -> W.MomentResult
```

L/B states are frozen scalar dataclasses. Numeric inputs must be represented
finite non-Boolean real scalars; response failures raise, not return a physical
zero. T>0; B mass>0, L mass>=0; 0<rtol<1e-4, tail30–100. B density inversion
requires strictly POSITIVE particle number. L density inversion allows signed
NET negatively charged-lepton minus antilepton number, including zero with a
nonempty pair plasma. Both inversion roots enforce relative number residual1e-9
for nonzero targets. These are numerical admissions, not accuracy envelopes.

Every material species has spin degeneracy TWO. T, mass and total chemical
potential include rest energy and have units MeV. Number, entropy (kB=1), and
Cv are fm^-3; energy, pressure and free-energy densities are MeV fm^-3.
Susceptibility `dn_dmu_fixed_temperature` and `dn_dtemperature_fixed_mu` have
fm^-3 MeV^-1; the returned fixed-number `dmu_dtemperature` is dimensionless.
L includes actual particle/antiparticle densities, rest-inclusive energy/P/s/f,
`cv_fixed_net_density`, and `dmu_dtemperature_fixed_net_density`. B includes
particle density, directly integrated kinetic energy, rest-inclusive energy/P/s/f,
`cv_fixed_particle_density`, and `dmu_dtemperature_fixed_particle_density`.
B has NO antiparticle density/diagnostic or mass-response field. Never relabel
particle number as net baryon number after silently adding an anti gas.
`thermal_energy_density` is a cold-subtracted L diagnostic, not the total energy
or the fixed-number derivative to use in the first law.

Package defaults MN939.565421, MP938.272088, ME0.510998950 differ from the weak
current's MN939.56542052, MP938.27208816, ME0.51099895069 MeV. Pass all three
campaign masses explicitly, including MP for the proton (B otherwise defaults to
MN). HC=197.3269804 MeV fm is common. Fix these as a named immutable binder model;
reject substituted masses. W.Current also fixes its bare weak mass
(MN+MP)/2=938.91875434 MeV and source-defined full Guo form factors. No vacuum Q
is added to material energy, and no GM1 common939/effective mass is substituted.

W accepts ONLY T1–3MeV, both baryon eta=(mu−mass)/T in[-100,-5], and |mu_e/T|<=3.
Radiation is exactly empty (all T/mu absent) or FD with positive T and independent
|mu_nue/T|,|mu_antinue/T|<=10. `channel_moments` additionally demands EXACT common
radiation/material T; it does not impose chemical LTE. Current/source functions
compute both actual FD branches. Channels are literal EC,PC,NEUTRON_DECAY.
Returned tuples are ordered(number, neutrino_energy, charged_lepton_energy,
proton_energy, neutron_energy, material_energy_loss), separately positive F/B and
signed net. Number units fm^-3 s^-1, energy MeV fm^-3 s^-1, entropy kB fm^-3 s^-1.
Positive finite logs with ordinaryNone mean under/overflow, not absence. Exact
inverse zero is permitted only for absent incoming neutrino occupation; signed
nonzero requires finite log_abs. Constructors alone are not proof a hand-built
record came from physical quadrature.

## 3. Proposed immutable public API and effective domain

The following are NEW proposed signatures, not existing functions:

```
state_density(temperature_mev: float, baryon_density_fm3: float, Ye: float,
              *, photons: str, controls: EOSControls | None = None
              ) -> FreeNPEState
source_moments(state: FreeNPEState, *, radiation: W.Radiation,
               controls: SourceControls) -> FreeNPESource
instantaneous_rhs(state: FreeNPEState, source: FreeNPESource,
                  *, baryon_density_dot_fm3_s: float) -> MatterDerivative
```

Use Python3.9-compatible typing spelling in actual code; the displayed union is
notation. Frozen EOSControls(rtol=2e-12,tail=60.0); frozen SourceControls contains
EXACT EC/PC/NEUTRON_DECAY W.Controls, with a `fine()` named constructor. Explicit
`photons` is only `absent` or `trapped_LTE`; no implicit thermostat/radiation
closure. Caller explicitly supplies W.Radiation; do not default equilibrium mu.
No integration or temperature inversion public API in this slice.

Initial rectangular admission is T1–3MeV,nB[4e-8,2e-6]fm^-3,.05<Ye<.9, then the
ACTUAL solved W bath must also pass its eta/radiation guards. Not every point in
that rectangle is admitted. The broader standalone EOS domains do not extend
rate coverage. Exact Ye0/1 and zero species remain rejected, consistent with B.
This follows prior research bounds rather than declaring arbitrary-degeneracy
rates. No uniform accuracy follows from request acceptance.

FreeNPEState contains immutable input/model/control identity, neutron and proton
B.ParticleState, electron L.LeptonState, explicit `muons_absent=True`, optional
analytic photon contribution, total energy/P/s/f/Cv, and e_n/e_Y/h_i responses.
Store no muon state: calling L with zero net muons would retain unwanted pairs.
Densities are nn=(1−Ye)nB, np=Ye*nB, electron NET=Ye*nB. Actual electron and
positron counts generally greatly exceed this net. No hidden beta equilibrium.

Proposed output records use these concrete fields (Python3.9 `Tuple` typing in
code): FreeNPEState has `temperature_mev: float`, `baryon_density_fm3: float`,
`Ye: float`, `photons: str`, `neutron/proton: B.ParticleState`,
`electron: L.LeptonState`, `muons_absent: bool`, `model: str`,
`eos_controls: EOSControls`, and `energy_density/pressure/free_energy_density:
float` in MeV fm^-3, `entropy_density/cv_fixed_nB_Ye: float` in fm^-3,
`denergy_dYe_fixed_nB_T: float` in MeV fm^-3,
`denergy_dnB_fixed_T_Ye: float` in MeV, `h_n/h_p/h_e: float` in MeV,
photon energy/P/s/Cv scalars and two immutable log anti-bound records.
FreeNPESource has `state_identity: str`, `bath: W.FreeElectronBath`,
`radiation: W.Radiation`, `controls: SourceControls`,
`channel_results: Tuple[W.MomentResult,W.MomentResult,W.MomentResult]` in literal
EC/PC/NEUTRON_DECAY order, `proton_number_source: W.SignedQuantity`,
`neutrino_energy_escape: W.SignedQuantity`, named n/p/electron_net/nue/antinue number sources,
material-energy loss/source, and explicit complete/provenance/ledger records.
Store original positive branches; a single aggregate sign does not erase them.
MatterDerivative has `temperature_dot_mev_s: float`, `Ye_dot_s: float`,
`baryon_density_dot_fm3_s: float`, and matched state/source identities. Only
represented finite source arithmetic produces this derivative; no ODE solver is
provided. Names above are proposed new names, not aliases claimed to exist today.

A separate log diagnostic bounds the OMITTED equilibrium anti-nucleon population:
f_FD(E,−mu)<=exp[−(E+mu)/T], hence nbar<=m²T K2(m/T)exp(−mu/T)/(pi²HC³)
for spin2. Evaluate its logarithm using scaled K2, and require bound/n_particle
<=1e-8 for EACH species before source admission. This is a proposed analytic
admission diagnostic, not an exposed B capability or a computed result of this
audit; unresolved bound rejects. It neither adds antibaryons nor estimates strong
interaction/model uncertainty.

## 4. Required energy and composition chain

Let **e(nB,T,Ye) be rest-inclusive ENERGY DENSITY** (also called u in some notes),
not energy per baryon. Set h_i=mu_i−T*(dmu_i/dT) at fixed species number; for e±
that fixed number is NET electron number. At fixed physical masses:

```
e = e_nucleon_n + e_nucleon_p + e_pair + e_gamma
h_n,h_p use B fixed-PARTICLE responses; h_e uses L fixed-NET response
Cv = e_T = Cv_n + Cv_p + Cv_e + Cv_gamma
e_Y = nB*(h_p-h_n+h_e)                  [MeV fm^-3]
e_nB = (1-Ye)*h_n + Ye*(h_p+h_e)       [MeV]
```

This uses Helmholtz d(e)/dn|T=mu−T*dmu/dT|n; chemical potentials alone are NOT
finite-T composition-energy derivatives. Photons, when declared, have
e_gamma=pi²T⁴/(15HC³),P=e_gamma/3,s=4e_gamma/(3T),Cv=4e_gamma/T and no fixed-T
composition/density derivative. f_total=e−Ts=sum component f−P_gamma.

For signed channel event-number nets r_EC,r_PC,r_D, define
R=r_PC+r_D−r_EC (net proton/electron creation), Q=sum signed neutrino-energy nets.
Then Q_material=−Q for this **included three-channel charged-current closure**, not a complete
free weak inventory. Evolution with imposed nBdot obeys

```
Yedot = R/nB
Cv*Tdot = [(e+P)/nB-e_nB]*nBdot - e_Y*Yedot - Q
```

Charge/baryon/lepton signs follow EC(+n,−p,−electron_net,+nue),
PC/decay(−n,+p,+electron_net,+antinue). Total electron-lepton number includes
nue minus antinue. Material entropy density per-baryon derivative is
[-Q−(mu_p+mu_e−mu_n)R]/(nB*T); material entropy need not increase with empty
incoming radiation. No total entropy claim follows from only escaping moments.
EM pairs stay in imposed pair equilibrium: their rethermalization can change
both particle populations while preserving electron net charge. This is a
mathematical closure, not a new EM thermalization/transport calculation.

## 5. Mandatory validation, provenance and future path scope

Factories perform fresh actual density roots and verify achieved species/net
numbers, temperatures, masses, summed identities, response positivity and finite
fields. Source admission canonically re-evaluates/checks component state from
T/nB/Ye and controls before constructing the exact W.FreeElectronBath. Reject
stale/altered component data or different source/closure/control identity; do not
trust a frozen dataclass or same-affinity label alone. All three source results
must be complete at EXACT same bath/current/radiation; timeout, invalid channel,
nonfinite quantity, unresolved positive branch or failed ledger aborts the
aggregate, never returns a first-channel partial as complete. Preserve all F/B
moments before signed sums. OrdinaryNone cannot enter an ODE arithmetic path:
raise an explicit unrepresented-source error while retaining the log record.
Reject spectral, opacity, fit, EOS-family, effective-mass and AD requests.

Installed execution must have no campaign paths, generic-module collisions,
driver imports, mutable current configuration or import-time file IO. Record
package revision, immutable model/constants and controls, process module origins
and available source identity; fixtures/build evidence additionally pin every
loaded component/helper hash. No persistent cache or saved-source substitution.
Provenance/guard admission is not an untrusted-code sandbox or empirical check.

Literal proposed **eight changed paths**, only after dependencies are integrated:

1. `src/nmbackend/reference/free_matter.py` (state/closure/response factory)
2. `src/nmbackend/reference/_free_matter_sources.py` (source sum and instantaneous RHS)
3. `tests/test_free_matter_reference.py`
4. `tests/test_free_matter_sources.py`
5. `tests/data/free_matter_reference_internal.json` (source-labelled INTERNAL rows)
6. `examples/free_matter_reference.py`
7. `docs/free_matter_reference.md`
8. `README.md` (small explicit-import/scope addition)

Public source/RHS functions may be reexported from free_matter with a local
function import to avoid a cyclic module dependency; do not alter package eager
exports or the physical component kernels. No RMF binder, spectral fallback,
source fit, ODE driver or new dependency enters this diff.

Acceptance: published/source identity first, independent momentum EOS forward
values before response checks; then finite-difference T/nB/Ye derivatives,
photon/pair identities, exact changing-bath EC/PC/beta forward profiles and tails,
positive F/B number/energy gates, rest-inclusive energy/Le ledgers on gross scales,
strict forged/stale/underflow/timeout tests, real-wheel imports and unchanged cold
regressions. A later evolution slice must independently resolve time and density-
work errors. Native AD, paired EOS, interacting matter, full inventory, broad
state coverage, calibrated uncertainty and full hydro remain outside this design.

## 6. Saved-arithmetic evidence, with execution attribution

`audit.py` imports only Python stdlib, statically extracts APIs and parses pinned
accepted records. Its120 checks pass; it performs ZERO new EOS/rate calls.
Four previously accepted states map actual root chemical potentials into W's
admission; unperturbed component forward agreement is checked before proposed
response arithmetic. At T2,nB1e-7,Ye.3, the saved EOS gives
mu_n919.8287578100662,mu_p916.8449303084382,mu_e0.17446333200826458MeV.
The later frozen actual event bath stores mu_e0.17446333200826397: the6.1e-16MeV
roundoff difference is explicitly tested, not presented as exact byte identity.
A future factory uses its ACTUAL solved value; no arbitrary substitution.

Independent previously executed empty-radiation event totals reproduce the
accepted binding R/Q within locked3e-6 gates. Saved arithmetic yields
R7.86475833838614e-7fm^-3s^-1,Q1.5155020638488612e-5MeVfm^-3s^-1,
Yedot7.864758338386141s^-1 and, with photons and prescribed nBdot2.5e-8fm^-3s^-1,
Tdot−1.7284560792876476MeV/s. These are reassembled SAVED moments/derivatives,
not a fresh physical chain, implemented binder or new trajectory. The existing
independent binder/trajectory reviews remain their own evidence. Saved method
agreement at the same Hamiltonian is not empirical physical-model uncertainty.
