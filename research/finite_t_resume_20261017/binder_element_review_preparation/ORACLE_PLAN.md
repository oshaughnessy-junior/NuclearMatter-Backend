# Candidate-blind short-element review preparation

STATIC ONLY: zero new EOS/source/phase integrations. The proposed author
binder_short_element/ directory has not been listed, read or imported. Only the
already independently accepted binder and its reviewed state/source semantics are
read. This preparation is not validation of a new trajectory or authorization to
spend the future source-call budget. Root must hand off the frozen author first.

## Physical closure and independent dynamical variables

Initial state T=2MeV,nB=1e-7fm^-3,Ye=.30,photons absent,muons absent, incoming
nue/antinue exactly empty. The density remains fixed for dt=2e-4s. Use the exact
accepted particle-n/p, NET-electron pair EOS and physical masses/current, not a
new EOS or stationary rate model. All three EC/PC/neutron-decay source results are
actual fresh events at each canonical stage. No saved rate or fixed-bath feedback.

The author proposes Euler2/4 in(T,Ye). The reference is explicit HEUN2/4 in
z=(rest-inclusive ENERGY DENSITY e,Ye). At each stage obtain T by a private scalar
root of F.state_density(T,nB,Ye,photons='absent').energy_density=e_target, checking
the achieved energy residual and actual rate-domain admission. Bracket[1.9,2.1]
MeV is predeclared for this short path; no clamping/extrapolation or automatic
bracket expansion. Failure is preserved as unresolved, not forced into a result.
The accepted binder's canonical roots and omission diagnostic run normally.

For every stage compute independent raw-channel sums
R=rPC+rD-rEC,Q=sum net neutrino energies. Then edot=-Q and Yedot=R/nB. This uses
no P*dn work because nBdot=0. Composition/rest-energy work enters ONLY through
EOS inversion when Ye changes; do not add eY*Ydot or an extra mass-Q to edot.
Do not advance with F.instantaneous_rhs: that would reuse the author temperature
chain. It may be checked later only as a named instantaneous identity, not as
another physical method.

For one step h:

1. Invert current corrected(e,Ye), obtain actual stage0 source k0=(-Q0,R0/nB).
2. Predict(e*,Ye*)=(e,Ye)+h*k0, privately invert, and obtain actual stage1 k1.
3. Correct(e_next,Ye_next)=(e,Ye)+h*(k0+k1)/2.
4. Invert the CORRECTED final state for T; a predictor source is not relabelled
   as a corrected endpoint source. Save both stage states and positive F/B
   moments/source identities, the root residual and their time weights.

Two steps require4fresh full-source groups; four steps require8. Maximum12groups
=36named event-channel evaluations if both runs are admitted. Each group is a
serial bounded process<=90s/thread1. It may check its own canonical state but
must not read author stored rates as reference. No extra endpoint/profile calls
fit within this budget. Initial physical source has to match the accepted binder
forward anchor before interpreting trajectory/time errors.

## Ledgers and independent error scales

Advance escaping energy, separate nue/antinue event counts, and the material
entropy integral with the same explicit Heun time rule. Independently reconstruct
these accumulators afterward from saved ACTUAL stage records using raw positive
F/B channel moments; do not trust only an aggregate identity or stepper's own
accumulator. Species-energy loss is separately assembled from p/n/charged-lepton
moments with the literal EC,PC,decay signs. This is an independent arithmetic
ledger at the same physical events, not new current-model accuracy.

Energy: e_final-e_initial+E_escape=0, normalized by gross escaped neutrino energy,
never e_initial~nB*939. Le: nB*(Ye_final-Ye_initial)+N_nue-N_antinue=0, normalized
by the integrated gross positive F+B event count. These near-algebraic ledgers
do not alone establish time accuracy; root residuals and genuine independent
method/time comparisons are mandatory. Charge/baryon signs remain explicit.

Material entropy per baryon has derivative
[-Q-(mu_p+mu_e-mu_n)R]/(nB*T). Compare its Heun integral to independently evaluated
EOS endpoint s/nB. Normalize by integral[|Q|+|mu_p+mu_e-mu_n|*grossR]/(nB*T),
not the rest-dominated entropy value or a cancelled net. Material entropy may
fall for transparent empty radiation. Number/energy moments do not supply a
radiation entropy functional or a claimed total second law.

Compare Heun2 versus4 and, after the author freeze, actual Euler4 versusHeun4.
LOCK.json gives absolute T/Ye plus change-normalized gates and escaped moment
time gates. Source-induced T/Ye changes are the error scales; small absolute
shifts are not automatically agreement. The source quadrature model/profiles
are held fixed. Time-method agreement cannot establish uniform source quadrature
accuracy or physical uncertainty; broader source-domain/refinement remains open.
Preserve all coarse failures and never tune gates after results.

## Resource handoff and partial outcomes

Fresh20:00:17UTC account/window/reset identity is exact but weekly use is96%,
versus root's earlier95. No new scientific batch starts at97%; hard stop99/reset,
identity drift or unavailable usage. No reset credits. Future review is conditional
on fresh<97 before each group and every<=10min, frozen author/source admission,
and explicit root handoff. At quota stop retain complete stages and unresolved
endpoints; Heun2 alone cannot be promoted to a fine/time certificate. A partial
plan/report is useful without fabricated missing stages or author-rate reuse.
This preparation requests no new public integrator, AD, long-time, full-inventory,
continuum transport or empirical model claim. It is deliberately a small controlled
reference comparison, not full hydro.
