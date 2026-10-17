# Independent review of the controlled Pauli-current slice

**ACCEPTED for the single locked free-bath current experiment only.** There
are329 current/thermal/provenance PASS (184 point/AST controls and145 thermal
checks), zero failed physical gate. Counts include correlated identities, not
329 independent physics benchmarks. The149 author passes remain separate.
No author/upstream/current/default-fit bytes, coefficients or domain changed.

Reviewed configured_currents.py SHA
d745784b53dbb1f8c37d6f0e1a8d67a22f95adadc98def254a6bcbbe5c756fbd,
manifestdf5f59ec94248841e7cc2505776a66e50cfdd9feead131a8686cda8c13728165.
Read the precomputation LOCK, builder/generated-current provenance, both
raw baseline/switched profiles, current controls and note before new probes.
This is a fresh independent numerical/algebraic assessment inside the same
source-defined weak model, not an independently measured physical model.

## Exactly what is removed, including the Gordon convention

Each generated baseline function is AST-identical to its pinned original
scalar function after the documented GF/Vud namespace substitution. Each
switched function differs by exactly one f assignment,3.706/den→0. F1's
(1−4.706*t/(4Mweak²))/den, GA, GP, masses, phase space, FD/Pauli factors
and all other terms are unchanged. Setting the common xi/Sachs parameter to
zero everywhere would also alter F1; my nonzero-t checks explicitly distinguish
that different intervention. No such refit is made here.

The declared convention is metric(+---),q=pb−pa in ordered inverse
absorption,sigma^{mu nu}=i[gamma^mu,gamma^nu]/2 and Mweak=(ma+mb)/2.
On shell, the unequal-mass Gordon identity is

`ubar_b i sigma^{mu nu} q_nu u_a = ubar_b [(ma+mb) gamma^mu−(pa+pb)^mu] u_a`.

Therefore the vector-plus-Pauli vertex between on-shell spinors is
`(F1+F2) gamma^mu−F2*(pa+pb)^mu/(2Mweak)`.
My independent spin-sum implementation uses THIS Gordon form, without the
source's sigma commutator or scalar invariant expansion, plus the unchanged
axial and pseudoscalar terms. At new nu/antinu points Enu6.3,23.1,71.2MeV,
cosines−.44,.36,−.82, both at rest and after common velocity(.13,−.08,.05),
it agrees with direct Dirac traces to7.78e−16 and with both configured scalar
amplitudes to1.00e−15. All amplitudes are positive. This verifies that the
switch does not mean removal of recoil or all momentum dependence: even in
the Gordon form it removes the linked F2 Dirac/convection contributions.

The direct trace's initial-baryon spin average2 is divided by2 for the
reciprocal charged-capture initial-spin average4. The independent spin sum
uses GF²*Vud²/8. The chiral projector retains the active emitted helicity;
no extra neutrino spin/DOS factor is applied to phase-integrated moments.
Baseline absolute thermal normalization is gated separately below.

At all new points, the Pauli vertex is q-transverse, its switched value is
exactly zero, V/A/P vertices are unchanged, and the full Dirac-adjoint
reversed-q identity passes. Only the Pauli component is asserted transverse;
unequal-mass full weak-current conservation is not imposed.

## Fresh actual EC/PC source profiles and forward gates

The bath is exactly the locked T3MeV,nB1e−7fm^-3,Ye.5490065797957709.
I independently solve actual particle-baryon FD densities by32-term scaled
Bessel fugacity roots, checked by original-momentum integrals, and neutral
pair-electron number by original-momentum Gauss integration. These pure EOS
functions are extracted from the earlier independent review helper without
importing its driver or source modules. New EOS chemical potentials differ
from the accepted adaptive roots by at most1.73e−15MeV. This is numerical
EOS-route independence; the free masses and physical EOS family are shared.

Every variant/profile has a fresh child process and fresh EOS. New controls
are EC24/40/10/10/10 tail55→30/50/12/12/12 tail70 and PC30/48/10
tail55→34/54/12 tail70. Both baseline profiles use the ORIGINAL current
functions and pass the unchanged3e−6 absolute forward gate against accepted
state_1.json BEFORE any F2zero child launches. Maximum baseline discrepancy
is1.83e−10 across all six separate moments. For the switch, only the owned
functions are injected in those fresh children. Each particle FD density
passes3e−8; each channel's rest-inclusive species/direct energy ledger closes.
All six moments refine within1.22e−10 for both variants, against3e−6.
No current normalization, density rescaling, inverse-KMS replacement,
stationary-target substitution or gate adjustment enters the comparison.

Fresh selected results (number fm^-3s^-1; energy MeVfm^-3s^-1):

| Channel/moment | Full baseline | F2zero | F2zero/full−1 |
|---|---:|---:|---:|
|EC number|5.010099210059192e−6|4.763749866209826e−6|−.0491705520|
|EC nue energy|7.318736618959669e−5|6.892283477173832e−5|−.0582686827|
|PC number|5.010072918779019e−6|5.273642674642003e−6|+.0526079680|
|PC antinue energy|7.811780799099808e−5|8.297870371122620e−5|+.0622251935|

These reproduce the author's conditional removal signs/sizes. They are
specific integrated moments at this bath, not all-energy sign theorems.
EC is p+e−→n+nue and PC is n+e+→p+antinue. Rest energies and active helicity
are already included; no extra reaction-Q or pair-rest term belongs here.

## What this says about the stationary allowed model

Using the immutable previously assessed allowed moments at the SAME bath,
allowed/F2zero−1 is EC number/energy+1.807997%/+2.579840% and PC number/
energy+4.871025%/+6.213826%. These allowed moments are saved independent
historical evidence, not newly integrated in this current-only experiment.
Their original assessment failures remain failures.

For PC energy, allowed/full−1 remains12.823002%, while F2zero/full−1 is
6.222519% and allowed/F2zero−1 is6.213826%. Ratios multiply:
`allowed/full=(allowed/F2zero)*(F2zero/full)`; percentages cannot simply be
added. The remaining allowed-model discrepancy is not isolated by this test.
It contains joint kinematic, distribution, vacancy and other-current changes.
Current interference and the fixed form-factor convention prevent treating
this slice as a unique additive physical-observable decomposition or medium/
empirical uncertainty estimate. It does not justify attributing the entire
original PC discrepancy to weak magnetism.

## Process isolation, failures and reproduction

The review thermal parent imports only standard orchestration modules; it
never loads mutable physical current/EOS modules. Four distinct child PIDs
have the same recorded parent PID. Switched amplitudes exist only in those
children; output JSON is written after both channel integrals and final
pin checks complete. The two successful baseline children and their profile
gates finish before either switched child. The author run_study.py likewise
uses serial sys.executable subprocess workers and baseline-first gates, with
no physical imports in its parent. No persistent parent global or upstream
file mutation supplies the switch. All155 author/source/fit-default paths
have identical before/after hashes, including the optional fit's full frozen
evidence and coefficients. Configured-current/worker execution identities
remain distinct from the unmodified phase-source labels in raw outputs.

Exact commands from Genesis, existing Python environment and no installs:

```sh
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/review_current_correction_assessment/current_probe.py
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/review_current_correction_assessment/thermal.py
```

The full four-child thermal experiment takes10.18s; all experiments are below
120s. An initial review-probe syntax error (missing whitespace in a Python
conditional) is preserved as initial_current_probe.py/.stderr/.stdout.
Correcting only that review code allowed the unchanged physical gates to run;
no physical/admission gate failed or was relaxed. All author149 checks,
previous physical coarse failures and allowed-model failures remain intact.

Accepted scope excludes beta/net-source/root claims, fitted-default changes,
muon channels, complete inventories, neutrino transport, thermal/hydrodynamic
evolution, interacting/mean-field/RPA, Coulomb/radiative corrections, broad
parameter-domain certification, physical uncertainty and derivative/AD claims.
No production/paper/git/PR operation or author fix was made.
