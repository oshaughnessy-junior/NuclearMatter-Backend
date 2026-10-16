# Finite-range incident-rule convergence, with explicit source succession

The locked24→32incident-energy rules pass3e-6 for every positive forward and
backward number/energy moment in all three declared channel ranges and radiation
closures. The preceding16→24rules also pass. This is incident-rule convergence
at the FIXED detailed inner profile and one exact free chemical bath, not a
whole-energy reconstruction or a total numerical/model error certificate.
DIFFERENT independent study review remains required.

| Channel | Lab energy range MeV | Worst16→24positiveR/Qchange | Worst24→32positiveR/Qchange |
|---|---:|---:|---:|
| EC |2–6|3.56e-15|3.56e-15|
| PC |2–6|2.86e-10|1.16e-12|
| neutron decay |.2–.8|9.31e-8|5.31e-11|

All29individual coarse failures remain visible:9PC4→8positive-moment checks,
and20beta4→8/8→16checks. Worst coarse changes are2.503e-5 for PC and8.206e-3
for beta. No gate, range, state, source formula, incident map, inner profile or
radiation prescription was retuned. Stop at32was respected. Passing finite-bin
conservation or source-affinity algebra did not erase the coarse failures.

## Locked physical and numerical problem

`LOCK.json` SHA657ace7fcd48d712a7a961769400bed05a2a45bd9b47509a7131bae615740716
was written before study physical calls. Bath T2,mu_n919.8287578100662,
mu_p916.8449303084382,mu_e.17446333200826397MeV is explicitly prescribed.
There is no EOS/neutrality/beta root, effective-mass model, rate fit or stored
moment substituted for detailed physics. The fixed physical n/p/e± masses,
full Guo VA/F2/GP current, rest energies, reciprocal transfer orientation and
normalization are those of the reviewed StageBsource. Baryons are particles
only; muons/meanfields/correlations/radiativeCoulomb are absent.

Exact fine controls: capture lepton12,angle16,baryon-energy16,phi6,inner-tail70T,
outer-tail40T; beta parent12,cone16,electron16,phi10,tail70T. Incident Gauss
orders4,8,16,24,32are the ONLY levels. Each channel has84actual coefficient
nodes; total252. Each node independently computes actual positive j and TRUEchi
with no external radiation occupation and no KMS-constructed inverse. Source
pin checks bracket every call. Jobs have≤4nodes, preserve incomplete status
until every node finishes, and use119sparent process timeout/110sAPIbudget.
Original EC/PCbatch elapsed204.49/192.19s; beta60.86s, all serial/thread1.
Every individual scientific process was below120s. No installs/production/Git/
paper/plot/dependency changes occurred.

The same actual node coefficients are explicitly reused only to apply:
exactly empty radiation; common-T FD LTE; and common-T changed-affinity FD
with EACH flavor chemical potential shifted−.8MeV from its own equilibrium.
Here mu_eq,nue=−2.809364169619812 and mu_eq,anti=+2.809364169619812MeV.
Thus changed mu_nue=−3.6093641696198118 and mu_anti=+2.009364169619812MeV.
These explicit numbers, not matching scenario labels, define the common bath.
No old PC/beta nonequilibrium labels with different mu are silently combined.

Positive branch moments are sums of one-helicity D(E)=E²/(2pi²HC³) times
j(1-f) or chi*f and positive incident weights; Q adds E. Number units are
fm^-3s^-1 and energy MeVfm^-3s^-1. The relative comparison is
abs(expm1(logM_orderA−logM_orderB)), separately for BOTHpositiveF/B R/Q.
Exactly absent empty-radiation B is structural zero and receives no relative
nearzero test. No cancelled net or baryonrest energy is an error denominator.
Nets use paired stable differences, with LTE normalized by positive gross.

## Actual source succession, not relabelled execution

The first168EC/PCnodes were physically executed on initial candidate18da0dfe...
and remain byte-for-byte in `raw/`, with their ORIGINALsource/worker/LOCK hashes.
Before the guard repair they were checkpointed in PARTIAL_CHECKPOINT.json and
ORIGINAL_SOURCE_CHECKPOINT_MANIFEST.json. All122initial checkpoint paths are
still unchanged. Their nine coarse failures remain unchanged.

The DIFFERENT initial candidate review found TWO record-admission problems,
not physical coefficient errors. The repaired candidate dcfbafe8... received
254passing independent rechecks; its physical current/phase/evaluator bytes
and canonical arithmetic AST remain unchanged, and all168study payloads
remain identical. The reviewed identity and exact frozen snapshot mapping are
in SUCCESSOR_SOURCE_ADMISSION.json SHA6c674e4a60b066f4ab42ec3acc698e9b8dfce85d33a4a13a61f24714e20fc89a,
written BEFORE any successor calls. Original LOCK/pins/data were not rewritten.

Beta's84nodes and all six fresh public API comparisons were executed on the
REPAIREDguard source, pinned in SUCCESSOR_SOURCE_PIN.json and `raw_successor/`.
The reports identify `original_18da0` versus `repaired_guard_successor` per
channel. Original EC/PCexecution does not become repaired-source execution.
This formal guard-only physical identity admission does not independently
qualify the present incident-convergence study; that review is still pending.

## Postprocessing and independent checks within this author study

Own harness output is explicitly labelled, never forged as an official
ReconstructionResult. Complete nodes/rules/indices, positive logs/ordinary
representation, source/worker/state/control/current/units and actual coverage
are checked before reuse. Hashes of raw chunks cover the SAME bytes parsed.
A changed source causes a stop. Twelve synthetic-copy attacks plus a valid
copy verify rejection of incomplete/missing nodes, wrong rule/state/control/
source/writer/current/units, falseempty, Boolean and nonfinite coefficients;
actual raw data are untouched.

Six FRESH public four-node fine-profile reconstructions (three channels,
empty andLTE) agree with own postprocessing:87checks pass, maximum1.83e-15.
They independently rerun24actual coefficients. This verifies same-node source
arithmetic and grouping, not four-node incident convergence: the known coarse
PC/beta failures remain. Original EC/PCreports and successor reports give
identical physical moment payloads;211source/metadata arithmetic identities
pass. Actual KMS, ordinary DOS arithmetic, LTE/gross, source-affinity entropy
and charge/baryon/Le bin bookkeeping are separated from incident-rule tests.
Final analysis records880PASS and29retained coarseFAIL; counts overlap checks
of the same fixed physical model, not numbers of independent models.

Per net forward ECevent material(n,p,e_net)=(+1,-1,-1),nue+1; PC/beta use
(-1,+1,+1),anti+1. Same-bin material lost energy is E times net emitted number,
without extraQ. These conservation/entropy expressions are algebraic checks,
not independent spectral species-energy closure. Empty incoming radiation has
no finite initial radiation entropy derivative. Fixed prescribed material and
FD radiation do not constitute an evolved closed thermal box.

Initial author base-profile2/4pilots with.34/1.71/5.29% sensitivities are copied
unchanged in prior_author_pilots/. They were explicitly unqualified pilots and
are not relabelled original failed certifications. Never mix their base inner
rules into the new fixed-FINE incident sequence. Reporting-only syntax errors
and the initial NaNnegative-input report encoding are preserved separately;
no physical data/source/gate changed to correct those reporters.

## Limits

All omitted low/high incident-energy ranges and whole R/Q/tails remain OPEN,
including the nonzero moving-neutron spectrum above the studied upper edge.
These selected finite-range incident differences do not certify inner
quadratures everywhere, continuum transport, an angular redistribution kernel,
arbitrary chemical states, signed net/root accuracy, EOS/evolution, AD,
complete opacity/weak inventory, external/empirical physics uncertainty or
minimum-runtime/fullCI performance. This bounded result can guide a later
explicit few-group configuration, but does not by itself qualify one.
