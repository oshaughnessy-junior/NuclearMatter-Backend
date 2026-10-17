# Staged free-npe material/source binder: initial review candidate

This candidate implements the previously reviewed free_matter_source_binder_contract,
using a real `nmbackend` package with the explicitly staged PR51/52/54 prerequisite
union. The new diff has exactly eight paths; 85 other baseline files are unchanged.
Existing main, draft branches and all prior frozen campaign files were untouched.
Independent physics, numerical and package review of this NEW adapter is pending.

The scalar CPU API provides immutable physical free-npe states, actual total
EC+PC+neutron-decay sources, and an instantaneous prescribed-density first law.
There is no time integrator, public T inversion, source interpolation/cache,
PR55 spectral surrogate, interacting matter or complete weak inventory.

## Model and first law

The package documentation in staged/docs/free_matter_reference.md gives the full
API/units/admission. Temperature T is MeV with no rest offset; e and mu include
rest energies. Neutrons/protons use physical masses939.56542052/938.27208816MeV
and particle number; e−/e+ use .51099895069MeV and net electron charge Ye*nB.
Yp=Ye, muons absent. Photons are explicitly absent or trapped_LTE. Baryon
antiparticles are omitted only after a separate log-scaled spin2 K2 MB density
upper bound/particle density <=1e-8 for EACH species. No extra Q is added.

Genuine fixed-particle baryon and fixed-NET electron thermal responses give
h_i=mu_i−T*dmu_i/dT|number, e_Y=nB(hp−hn+he),
e_nB=(1−Ye)hn+Ye(hp+he). With R=PC+decay−EC and Q the sum of net neutrino
energy moments, Ydot=R/nB and

```
Cv*Tdot = [(e+P)/nB − e_nB]*nBdot − e_Y*Ydot − Q.
```

Original separate forward/backward positive/log moments are retained, including
p/n/charged-lepton energies and direct material losses. Their independent energy
and charge/Le ledgers must pass BEFORE assigning the included material source−Q.
The entropy chain is material-only: d(s/nB)/dt=[−Q−(mup+mue−mun)R]/(nB*T).
Empty-radiation entropy derivatives are not asserted.

Admission requires the actual solved PR54 chemical bath, not just the rectangular
T1–3,nB4e−8–2e−6,.05<Ye<.9 bounds. Canonical EOS roots and full state/control/
source-file identities are recomputed before source and RHS admission. Runtime
source pins cover all seven EOS/weak prerequisite files plus constants, and binder
hashes/module origins are recorded. The RHS receipt additionally rejects altered
or reconstructed source records; it does NOT replace canonical roots or promise
security against arbitrary Python memory modification. A source with an ordinary
underflow/overflow remains log-bearing, but the RHS rejects it rather than treating
it as zero. No persistent source cache is present.

## Completed author evidence

Forward values precede derivatives. All experiments use one numerical thread and
completed in under14s per scientific process (well below90s budget).

- 72 checks: independent momentum-space n/e± number, total energy, pressure and
  entropy integrals at three states, followed by decreasing-step density-root FD
  checks of Cv,e_Y,e_nB. Component production quadrature uses kinetic-energy
  coordinates. Largest response relative difference3.86e−6 against1e−4 gate;
  numerical method agreement is an internal check, not physical-model accuracy.
- Actual full three-channel sources at THREE freshly solved states, with no saved
  source substitution. Original full channel results/provenance are in source_0/1/2.json.
- 84 checks: independent ordinary channel sums and p/n/e energy ledger, explicit
  Le signs, first law, independent momentum MB anti bound (log differences <=
  2.28e−13), and fresh FD entropy-response step ladders. Largest entropy-chain
  relative difference1.04e−5 at h=.002, falling to2.60e−6 at .001 (gate1e−4).
- 19 checks: separately computed BASE versus FINE full forward number/energy/
  charged/p/n/material moments at the CHANGED second state; largest relative
  difference5.21e−9 versus declared3e−6 gate. This is selected convergence only.
- 25 package tests: canonical state/control/provenance guards, explicit photon/
  no-muon closures, anti-bound rejection, malformed/channel-missing/timeout source
  failures, altered state/source and stale prerequisite pins BEFORE source/RHS,
  and finite-log positive underflow that rejects ordinary RHS. Source fault
  injection uses saved ACTUAL channels solely to exercise assembly/guards; it is
  explicitly synthetic and supplies no new physics evidence.
- Offline existing-toolchain wheel build, unpacked without installing dependencies,
  imported from /private/tmp in a fresh process; all25 tests pass against wheel
  modules. An ACTUAL wheel full3-channel source/RHS matches staged source0 within
  2e−12 (3 checks). Exact module origins and environment in wheel_result.json.
  This is not clean dependency resolution, minimum-version CI or external oracle.

Actual values (R fm−3 s−1, Q MeV fm−3 s−1, nBdot=2.5e−8fm−3 s−1):

| T,nB,Ye; photons; neutrinos | R | Q | Tdot MeV/s |
|---|---:|---:|---:|
|2,1e−7,.30; absent; empty|7.864758338386145e−7|1.515502063848861e−5|−2.782499683617574|
|2.03,1.05e−7,.34; trapped_LTE; empty|7.404324794184863e−7|1.693834242722024e−5|−1.885841264545582|
|1.7,2e−7,.21; absent; FD(T,mu_nue=−1.3,mu_anti=2.1)|9.171932151797207e−7|1.021155990225391e−6|−.161644980202385|

Changing composition/T/density changes the evaluated bath and actual sources.
These are instantaneous configurations, not trajectory or beta-equilibrium claims.
Source0 R/Q also match the old contract's selected total-source anchors. Photon
closures differ from that note's trapped-photon Tdot, so its Tdot is not reused.

## Preserved failures and open gates

Initial code/results before stricter canonical diagnostics/source completion are
preserved in initial_guard_completion/; final source0/1 were physically recomputed,
source2 and final EOS/ledger reports use final code. Two completed ancillary probes
initially failed only JSON serialization of numpy.bool_; original scripts/failure
records are preserved, explicit Python-bool recorders fixed and probes rerun.
The diff generator initially used the wrong baseline-parent path; it failed before
writing a diff, then passed with verified baseline hashes. No failed physical
threshold was relaxed or hidden. There are zero failed final author scientific
checks. No uncompleted calls are counted.

The prerequisite PR54 model/error/domain limits remain: selected dilute free matter,
full-current prescribed form factors, no muons/scattering/pair processes, nuclear
interaction/radiative/Coulomb/medium corrections or empirical uncertainty envelope.
Constructor admission is not a uniform quadrature certificate. Native AD, broad
parameter/state accuracy, clean CI and independently reviewed package ownership are
OPEN. Runtime exact file pinning intentionally requires a conscious contract update
for future prerequisite changes. No external EOS validation or full cooling/hydro
qualification is claimed.

Review entry points: staged/src/nmbackend/reference/free_matter.py and
_free_matter_sources.py; exact proposal PROPOSED_DIFF.patch; prerequisites
PREREQUISITES.json/STAGED_BASE_PIN.json; numerical scripts and raw JSON beside
this note. No network/publishing action is included in this leaf.
