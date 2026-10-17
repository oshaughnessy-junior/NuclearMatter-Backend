# DIFFERENT short-element review: energy-variable Heun reference

**Restricted Heun self-consistency and selected endpoint agreement accepted.
The author's Euler2/4 energy accuracy FAILS the independent prelocked3e-6
escaping-energy ledger criterion.** Both failures remain failed. The author17PASS
record uses a different, looser2% criterion; it is not wrong source arithmetic or
a new EOS/model discrepancy. This review does not report “all candidate accuracy
gates pass,” an asymptotic order certificate, or a public ODE API.

## Candidate-blind plan and execution

Before author access, binder_element_review_preparation/LOCK.json
a647ebe709c66bd0ee92743028f7b8a9efaed4ade18977c811c75e6f1103dee3 and
ORACLE_PLAN.md be5db1c5e53e97afbc8602918650b66049646f52fc8efa9f2e6258a36ccdefa1
fixed this independent comparison. Root then admitted the frozen author manifest
aa43c8eab89d3492123c022d28a79e80e1be8818558529c5ea1cb6739bacdd79 and
driver195fc3e0c4b094811bdbd95522a7c293c6dfad8f36ef35fa5a47c832d06b8847.
No new author/source files were changed. All author members and the126 accepted
binder members were checked before/after each scientific call and at freeze.

The physical problem is exactly initial T2MeV,nB1e-7fm^-3,Ye.30,photons absent,
muons absent, empty incoming nue/antinue, duration2e-4s and fixed density. Material
remains the declared free FD/pair equilibrium ansatz, with physical masses and
actual EC+PC+neutron-decay/inverse total-event sources. This is an included
three-channel closure, not a complete weak/EM inventory.

The author uses Euler in(T,Ye). This review instead implements Heun2/4 in
rest-inclusive ENERGY DENSITY e and Ye, edot=-Q,Yedot=R/nB. Every current/predictor
stage independently inverts the canonical EOS for T before fresh source calls.
No author stored rate is reused; no fit/interpolation, pressure work, additional
eY*Ydot term or mass-Q enters edot. Composition/rest-energy changes enter through
EOS inversion only. The stage0 physical source matches the previously accepted
binder R/Q before trajectory interpretation. This uses the same reviewed physical
current/event/EOS model, not a second interaction model or empirical calibration.

All12 planned full source groups (36 event-channel calls) completed serially,
thread1, in about8.5–11.1s each under89s alarms. Each group had its own freshly
saved time/quota identity check:96% weekly usage, original account/codex10080/reset
1792652325, below97% wind-down. No call was started at97%, no reset credit was
consumed, and no extra source/profile/endpoint calls were made. Each corrected
endpoint has a fresh EOS inversion but no source is claimed there unless actually
evaluated. Predictor sources are explicitly labelled as predictors.

## Independent saved-stage arithmetic and numerical results

After calls, a separate stdlib-only analysis reassembles raw channel net number,
original positive F/B gross scales, neutrino energy and p/n/charged species-energy
loss with the actual Heun time weights. Escaping energy, separate nue/antinue
counts and material entropy are reconstructed independently of stored stepper
accumulators. These are arithmetic checks at the same events, not independent
weak-matrix-element accuracy. EOS endpoint energy/entropy is actually evaluated;
it is not defined from a ledger.

| Method | Final T [MeV] | Final Ye | Escaped E [MeV fm^-3] | Energy defect / gross escape |
|---|---:|---:|---:|---:|
| Author Euler2 |1.999409520470466|.3015699483091133|3.028968896900208e-9|2.508151525e-4 FAIL|
| Author Euler4 |1.999409626300147|.3015684519688716|3.027953489378919e-9|1.253734751e-4 FAIL|
| Independent Heun2 |1.999409732156538|.3015669564302135|3.026938474592781e-9|6.479163890e-10 PASS|
| Independent Heun4 |1.999409732024835|.3015669584851011|3.026939278795149e-9|−2.311528399e-10 PASS|

The independent energy gate is3e-6 for all four rows, unchanged from preparation;
no tolerance was relaxed after seeing the Euler defects. The two absolute Euler
energy defects7.5971e-13/3.7963e-13MeVfm^-3 are small but remain too large on the
specified escaping-energy scale. Normalizing by e_initial~nB*939 would conceal
that distinction and is not used. Their halving is a measured two-rung trend,
not enough to establish an asymptotic order or qualified longer-time trajectory.

Heun2→4 changes final T by1.32e-10MeV and Ye by2.06e-9. Escaped energy and
individual flavor counts pass the prelocked3e-5 time-sensitivity gates. Euler4
versus Heun4 differs by about1.06e-7MeV and1.494e-6 in Ye, passing the separate
2e-6 absolute and2e-3 source-induced-change-normalized endpoint gates. Endpoint
agreement DOES NOT waive the independently failed Euler energy criterion.

The independently reconstructed Heun lepton ledger residual is<=1.19e-14 of
gross emitted event count. This checks nB*deltaYe+N_nue-N_antinue, not a nearzero
net denominator. Material entropy endpoint-minus-integral/gross scales are
−2.95e-8 and−7.72e-9, below1e-4. Its derivative is
[-Q-(mu_p+mu_e-mu_n)R]/(nB*T), and material entropy decreases by about.01294 per
baryon. Empty-radiation moments do not define a radiation entropy functional or
an asserted total second law. Baryon density is prescribed constant; there is
no P*dn contribution or density-driven work being validated here.

The analysis contains137PASS and2FAIL (both Euler accuracy), with no incomplete
rung, skipped source node, source underflow or budget-stopped call. All actual
root energy residuals satisfy the locked stage-energy/absolute criteria; the T
bracket remained[1.9,2.1]MeV without clamp or expansion. Actual source/radiation/
control and current identities, complete channels and predictor/corrected roles
are recorded. No physical or numerical gate was changed; there are no reviewer
harness failures in this short-element batch.

## Disposition and limits

The selected Heun reference and endpoint comparison are useful controlled
integration evidence. The author's Euler2/4 demonstration remains **unqualified
for the stricter review energy-accuracy target**, while its original own-gate
17PASS report is preserved literally. This does not revoke the accepted underlying
instantaneous binder/current/EOS component review or identify a model error.
Improved Euler accuracy, a different production integrator or longer trajectory
requires a separately scoped test, not a silent repair or reordered gate.

Heun e-ledger conservation is partly structural; it cannot substitute for time
refinement, independent current/phase checks, source-tail convergence at every
changing state, or empirical physics uncertainty. Source controls are held at
the accepted fine model; this review adds no independent quadrature certificate
for all intervening baths. Two rungs do not prove continuum time order. No hydro,
opacity/spectral, full-inventory, paired/interacting EOS, AD, long-time evolution,
minimum-runtime/full CI or empirical-model qualification follows. No further
scientific calls are requested. Preserve all successful raw stages and both
accuracy failures in publication/archive.
