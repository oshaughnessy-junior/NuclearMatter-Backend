# Independent selected fitted-source root assessment

**ACCEPTED for the two locked research assessments only:182 passing checks,
zero failed numerical/model/ledger gates.** These checks include correlated
moments and identities, not182 independent physical benchmarks. The actual
detailed source remains nonzero at both fitted roots. No fit, coefficient,
domain, source, author file, production file or earlier review was changed.

Reviewed assess.py SHA2d591cfe2c0687d068c14f3f38c9ca19144688d1b6c4a2b391f32fc824639b2d,
manifest196f665fd97612247a79d68d9d95193ac90086e1cbf226de7c0e67ffdc6a6ac6.
Read the pre-source LOCK, actual source, raw trials/profiles and note, and the
accepted fit-consumer repair and physical-balance reviews before experiments.
The author112 numerical/4 model passes remain separate from this review.

## Independent EOS and newly solved approximate roots

I independently invert relativistic particle baryon densities using the32-term
FD fugacity/scaled-Bessel series, retaining Pauli corrections. Original
momentum integrals separately check those baryon densities and provide energy
and fixed-T energy-per-number response. Neutral e−/e+ number, energy and
susceptibility are integrated over original momentum with80-point panels;
112-point refinement is checked. This differs from the author's adaptive
kinetic-energy EOS. Actual source masses and HBARC are retained; no Maxwell
substitution, baryon renormalization or common electron/muon equilibrium is
introduced. Muons/antibaryons are absent in the original selected model.

At EVERY independent root trial the EOS is solved, the unchanged optional
positive EC/PC API is called, and the actual full-current beta source is
integrated afresh. No saved beta/source value enters root feedback. Each bath
has8 trial evaluations using a new beta profile20/26/10/14, pair rest/direct
parent coordinates, tail75. New brackets are the candidate composition±.0008;
the left fitted source is positive and the right negative. A separate brentq
solve gives:

| T MeV,nB fm^-3 | Independently solved fit+actual-beta Ye | Delta from accepted detailed root | Fresh detailed net/gross | Actual dYe/dt s^-1 |
|---|---:|---:|---:|---:|
|1,1e−7|.38505642322690625|−2.8696889737e−7|+1.1519507767e−6|+4.4491957173e−7|
|2,1e−6|.27862711849685773|−6.3937326078e−7|+3.1267209129e−6|+3.7149088763e−5|

Both new roots differ from the author's by less than1e−14 in Ye. Both locked
|deltaYe|<=.001 and actual |net|/gross<=.001 criteria pass independently.
The approximate root's numerical net/gross gate is1e−9, distinct from model
bias. No relative error against zero detailed net is used. The saved accepted
detailed roots are immutable reference evidence; a new multidimensional
physical root search was not run. Neither global uniqueness nor a general
net-rate/root API follows.

Original outside-domain roots at T2,nB1e−7,Ye.5560448218 and T3,nB1e−7,
Ye.5490065798 reject through the unchanged fit API. They are not clipped,
extrapolated, retuned or included in the selected assessment. Source inspection
confirms the author's own approximate callback also prepares fresh EOS and
actual beta before returning each trial; raw records contain that data.

## Fresh detailed EC, PC and beta at the new roots

All three actual event sources execute at BOTH new roots with two separate
profiles: EC24/40/10/10/10 tail55→30/50/12/12/12 tail70; PC30/48/10
tail55→34/54/12 tail70; beta20/26/8/12 tail75→26/30/10/16 tail90,
electron rest/direct parent coordinates. All separate number, escaped-energy,
charged-lepton-energy, proton/neutron-energy and direct material-loss moments
refine within1.28e−10 against3e−6. Each incoming FD density passes3e−8.
The residuals above are retained with their signs and absolute dYe/dt.
Fresh versus saved detailed residual differences are compared absolutely in
gross-source units, not relative to a cancellation-conditioned small net.

Fit fractional errors in the individual R_EC,Q_EC,R_PC,Q_PC moments are,
respectively, (−7.648e−7,−7.702e−8,−3.074e−6,−2.560e−6) and
(+1.909e−6,+9.769e−7,−4.345e−6,−3.136e−6). They remain diagnostics,
separate from root displacement and gross-residual gates; no new individual
acceptance criterion is chosen after observing them.

This is numerical independence of EOS inversion/root arithmetic and fresh
source controls within the previously accepted physical model. It is NOT a
new independent current/tensor derivation or nuclear-model uncertainty test.
The PC kernel is the actual independently accepted scalar phase reference,
not saved moments. EC/beta use the qualified actual kernels; all currents,
form factors, weak masses, spin/helicity factors and empty incoming radiation
conventions remain pinned. No extra neutrino DOS/spin factor is added.

## Thermostat, rest energy and pair closure

Literal stoichiometry gives dn_n/dt=−R,dn_p/dt=R and net electron number
source R=Rbeta+RPC−REC. Escaped electron-lepton source REC−Rbeta−RPC=−R,
so baryon number, electric charge and total electron-lepton number close.
Separately accumulated species energy changes sum to−Qescape, while the
positive direct collision material-loss moment equals Qescape. All three
positive escaped energies are ADDED.
These moments already include rest energy; no extra neutron-proton Q or2me
pair-rest term is added.

From independent momentum FD response moments I obtain fixed-T composition
energy derivatives uY=1.00979536004297 and2.85599602624539 MeV. Finite
composition differences of the rest-inclusive total EOS energy at h=3e−4
and1.5e−4 agree to better than4.6e−7 MeV, against2e−5. This gates forward
EOS energies before using their response identity; it does not qualify AD.

External power is H=Q+uY*R, with the ACTUAL detailed R. The independently
reconstructed values (MeV fm^-3 s^-1) are:

| T,nB | Qescape | Required fixed-T H |
|---|---:|---:|
|1,1e−7|2.004575981718776e−7|2.004576430996495e−7|
|2,1e−6|1.205897054000380e−4|1.205898114976879e−4|

The composition terms H−Q are4.49277719e−14 and1.06097650e−10, retained
rather than silently setting R to zero. Substituting the independently
finite-differenced EOS derivative also passes the directional power ledger.

Microscopic e− source is Rbeta−REC and e+ source is−RPC. Projecting the
fixed-T neutral pair bath with independent susceptibility fractions gives
identical e− and e+ EM replenishment rates:1.92799248e−8 and5.94028137e−6
fm^-3 s^-1, both positive. EM pairs carry zero net electric/lepton number;
the reservoir and redistribution supply the energy already captured by H.
No photon evolution, closed-box thermal equilibrium, full inventory or cooling
ODE is inferred. Nonzero chemical affinities and detailed drift are consistent
with empty incoming radiation and a prescribed bath.

## Reproduction and preservation

roots.py gives85 PASS in20.59s; sources.py gives97 PASS in15.48s. Exact commands:

```sh
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/review_fit_balance_assessment/roots.py
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/review_fit_balance_assessment/sources.py
```

Both experiments use the existing environment, stay below120s, and preserve
full raw trial/profile/ledger values, stdout/stderr and before/after hashes.
The185 frozen author/fit/accepted-review/source paths verify unchanged; every
manifest and source pin is checked before new source work. The assessment/grid
drivers are never imported or run. Only explicitly loaded physical channel
modules and the accepted optional API are used. common.py is this review's
pure helper. No installs, source fixes, coefficients, production/paper/git/PR
operations or relaxed gates occurred. A discovery read found no optional
EXPERIMENT_PLAN.md; the actual preregistration is LOCK.json, which was read.
No scientific probe failed.

Acceptance is bounded to these two chosen baths and their specified practical
criteria. Detailed physical quadrature remains the reference/default; positive
individual-moment fits remain optional. Whole-domain source signs, uniform
root errors, physical uncertainties, alternative EOS/current families, other
radiation closures, transport, complete weak inventories, long-time dynamics
and derivative/AD claims remain excluded. All preceding allowed-model,
fit-admission and source-review failures remain historical evidence.
