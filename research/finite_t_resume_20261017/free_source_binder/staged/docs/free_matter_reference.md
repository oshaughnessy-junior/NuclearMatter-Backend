# Restricted free-matter and charged-current instantaneous reference

This CPU research adapter requires the exact finite_t_leptons, finite_t_baryons
and total weak_reference prerequisite union. It is not independently available
from main until those dependencies are integrated. Use explicit imports:

```python
from nmbackend.reference import free_matter as F
from nmbackend.reference import weak_reference as W

s = F.state_density(2., 1e-7, .3, photons='absent')
source = F.source_moments(s, radiation=W.Radiation.empty(),
                         controls=F.SourceControls.fine())
rhs = F.instantaneous_rhs(s, source, baryon_density_dot_fm3_s=2.5e-8)
```

No integrator, public temperature inversion, AD, EOS-family switch or spectral
opacity/fit fallback is supplied. The API is scalar and resource-bounded by the
explicit underlying quadrature controls; constructor acceptance is not a uniform
accuracy envelope. Current prototype qualification is selected internal states
and pending independent review, not empirical physical-model validation.

## Fixed model and state

T is temperature in MeV, with **no rest-energy offset**. Total chemical
potentials, energy density and free-energy density include rest energy. Fixed
physical masses are mn939.56542052,mp938.27208816,me.51099895069MeV and
HC197.3269804MeVfm, explicitly passed to both component EOS factories. They match
the total weak reference; no additional vacuum Q or common939 RMF mass is used.
Each material species has spin2. The weak-current bare mass and form factors are
fixed by its source, distinct from matter thermal responses.

nB is fm^-3, neutron particle number is(1-Ye)nB, proton particle number isYe*nB,
and electron **NET** number isYe*nB. Both actual e−/e+ populations remain present
and generally exceed the net charge. Nucleons are particle-only; no anti gas is
silently added. Muons are explicitly absent, so no zero-net muon pair state is
constructed. There is no nuclear beta-equilibrium assumption.

`photons` is mandatory: exactly `absent` or analytic `trapped_LTE`. The latter
adds eγ=pi²T⁴/(15HC³),Pγ=eγ/3,sγ=4eγ/(3T),Cvγ=4eγ/T with no fixed-T density or
composition derivative. This does not select incoming neutrino radiation.

Admission is T1–3MeV,nB4e-8–2e-6fm^-3,.05<Ye<.9 AND the actual solved weak bath
(both particle-baryon eta[-100,-5],|mu_e/T|<=3). The rectangle is not sufficient
by itself. Exact zero species, substituted masses and unsupported closures reject.
EOSControls is frozen, rtol2e-12/tail60 by default, with the exact component
numerical admission bounds. Fresh actual roots and response positivity gates
precede any use of the state. Canonical source admission recomputes those roots
and compares the complete state/model/control/provenance identity.

For each omitted anti-nucleon species the spin2 FD occupation is bounded by MB:
`log nbar_bound = log[m² T/(pi² HC³)] + log(kve(2,m/T)) - m/T - mu/T`.
The bound/particle-number ratio must be<=1e-8 separately for n and p. It is an
analytic omission diagnostic, not a computed anti-baryon EOS or strong-interaction
uncertainty estimate. A finite log is retained when an ordinary bound underflows.

## Thermal/composition chain and first law

With e denoting rest-inclusive **energy density**, h_i=mu_i-T(dmu_i/dT)|number.
The nucleon number is particle number; electron number is net charge. Actual
fixed-number Cv and these h_i give

```
e_Y = nB*(h_p-h_n+h_e)               [MeV fm^-3]
e_nB = (1-Ye)*h_n + Ye*(h_p+h_e)    [MeV]
Cv = Cv_n+Cv_p+Cv_e+Cv_gamma        [fm^-3, kB=1].
```

Chemical potentials alone are not composition-energy derivatives. Cold-subtracted
lepton thermal energy is not substituted for total energy or Cv.

The INCLUDED three charged-current channels are literal EC,PC,NEUTRON_DECAY.
All compute actual full event F/B totals at the canonical chemical bath; PR55
spectra, finite incident bands, opacity kernels, emulators and saved-source
substitution are not used. Original positive branches, signed/log nets, bath,
current, controls, radiation and diagnostics are retained in immutable records.
This is not a complete free weak inventory: it excludes scattering, pair sources,
muonic reactions and other processes.

For rEC,rPC,rD and summed neutrino-energy net Q, R=rPC+rD-rEC creates protons and
electron net charge. It destroys neutrons. Nue source=rEC; antinue source=rPC+rD.
Thus baryon, charge and electron-lepton ledgers close. Independent material-energy
moments are checked against neutrino escape on gross F/B scales, then the included
closure defines material energy source=-Q. Q can be negative with irradiation.

```
Yedot = R/nB
Cv*Tdot = [(e+P)/nB-e_nB]*nBdot - e_Y*Yedot - Q
d(s/nB)/dt = [-Q-(mu_p+mu_e-mu_n)*R]/(nB*T).
```

nBdot is explicitly prescribed, not solved by hydro. Material entropy need not
increase with empty radiation. No total radiation entropy claim follows from
escaping energy moments alone. e± pairs remain in imposed local pair equilibrium;
the adapter does not calculate EM thermalization or photon transport.

Radiation is explicitly `W.Radiation.empty()` or common-exact-T FD with independent
nue/antinue chemical potentials. No chemical LTE default is inferred. A timeout,
incomplete/wrong channel, nonfinite record, altered state/controls/current or
failed ledger rejects the aggregate; no partial source is marked complete.

Positive support retains its finite log and ordinaryNone for under/overflow. Such
a complete log-bearing source can be inspected, but instantaneous_rhs raises
UnrepresentedSourceError retaining that source rather than producing a false zero
or nonfinite ODE derivative. Factory receipts reject replaced/hand-constructed or
stale source records at RHS admission. They are same-process consistency markers,
not serialization capabilities, persistent caches or an untrusted-code sandbox.

## Source identity and limits

Runtime provenance records exact prerequisite/binder file hashes and module
origins, fixed model/constants and immutable controls. It performs file reads
on explicit factory calls, not binder import. Available package/stack revision
and environment identities are recorded by the staging study. Source equality
is not a proof of arbitrary Python-memory integrity or physical accuracy.

Selected independent momentum forward anchors precede response finite differences.
The research reports retain actual changing-state channel values, quadrature
controls, ledgers and failures. Normal full CI, minimum dependency versions,
independent physics/numerical/package review, native AD, paired/GM1 matter,
interaction uncertainty, broad-state accuracy and whole cooling/hydro evolution
remain separate work.
