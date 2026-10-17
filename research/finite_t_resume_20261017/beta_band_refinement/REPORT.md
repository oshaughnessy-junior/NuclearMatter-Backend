# Upper beta incident interval: preserved coarse failure, converged higher pair

Author evidence, 17 October 2026. This is a selected finite-interval quadrature study for [1.1,1.4] MeV only, using the unchanged frozen physical full Guo conditional neutron-decay/inverse kernel at the exact fixed T2 chemical bath. It makes no whole-energy, event-total or transport claim. DIFFERENT phase/incident review remains pending.

## Prelocked experiment and source identity

Before physical outputs, `LOCK.json` fixed Gauss incident orders 12,20,32; the exact generated nodes and weights; 72 maximum calls; current/source hashes; the baseline inner rule; two radiation states; the positive-branch gate 3e−6; and the policy that only the adjacent higher pair 20→32 may admit this selected interval. The 12→20 comparison must be reported and any failure preserved. No thresholds or gates were adjusted after seeing values.

The frozen `conditional_beta.py` candidate hash is `148c99c37dc0550ca983b53d86123e25125469b15fd2fde9edc16c169c2ac191`. The entire required source is copied under `frozen_beta_support/`, with explicit source names and byte-for-byte hash admission. No original source or previous output is imported through a generic module alias, edited or overwritten. The model remains free particle neutron beta decay and its inverse with physical masses, full named Guo VA/F2/GP current, active antineutrino helicity, actual separate FD products and exact future-shell support.

All 64 baseline nodes use parent radial Laguerre16, cone20, electron20 and phi14. The two edge nodes of the order32 rule are E=1.1004104207225778 and 1.3995895792774222 MeV. At each edge the four extra profiles separately vary radial order to24, cone order to24, electron order to24, and replace the radial integral by fixed parent-excess panels to tail90 while holding other controls fixed. This last control is an alternate radial-domain/rule comparison, not a claim of a rigorous isolated missing-tail bound. Phi isolation was not run.

Every bounded call finished below one second; all numerical threads were limited to one. Quota/account/window/reset were checked afresh before every batch of at most eight serial calls. Exactly 72 calls were used, with immutable per-node raw JSON. The 64 baseline coefficients were not cached from another study or replaced by source moments; only their reuse for two declared radiation weightings is arithmetic.

## Positive number and energy integrals

Let the frozen kernel return j(E) and true chi(E), independently calculated with actual event occupations. They are per-occupation s−1 coefficients, with no external radiation occupation or neutrino DOS inside them. For a chosen antineutrino occupation f(E), form separately

```
F_R = integral_[1.1,1.4] dE D(E) j(E) [1−f(E)]
B_R = integral_[1.1,1.4] dE D(E) chi(E) f(E)
F_Q = integral_[1.1,1.4] dE E D(E) j(E) [1−f(E)]
B_Q = integral_[1.1,1.4] dE E D(E) chi(E) f(E)
D(E) = E²/[2 pi² (hbar c)³].
```

The one-active-helicity DOS is applied exactly once. R is fm−3 s−1 and Q is MeV fm−3 s−1. The two radiation states are exactly empty f=0, and common-T FD with T=2 MeV and mu_antinu=0. Empty radiation's B source is structurally absent incoming occupation, while chi itself remains positive at all actual nodes. This is the only exact zero in the branch arithmetic.

All sums use positive log-space arithmetic. A positive finite log that underflows in ordinary float64 remains positive with ordinary representation None; it is never treated as absent support. No signed-net cancellation, gross sum, baryon rest energy, or weighted-smallness argument substitutes for the separate F/B R/Q comparisons.

## What fails and what passes

The original October 15 upper-panel failures are preserved with their exact prior source hash in `PRIOR_FAILURES.json`. The new order12→20 comparison also fails; it is not repaired by the better local kernel:

| Positive branch | Relative order20/order12 − 1 |
| --- | ---: |
| Empty F_R | +1.9788743e−5 |
| Empty F_Q | +1.8219919e−5 |
| FD(mu=0) F_R | +1.9470364e−5 |
| FD(mu=0) F_Q | +1.7921934e−5 |
| FD(mu=0) B_R | +1.9470364e−5 |
| FD(mu=0) B_Q | +1.7921934e−5 |

These six failures remain in `ANALYSIS.json`. The spectrum falls steeply through this upper interval, so the incident quadrature itself needs refinement even after the local conditional support repair. The interval's small contribution to a wider source integral provides no waiver.

The prelocked higher pair order20→32 passes each of these positive branches at relative differences no larger than 5.69e−14. Edge controls separately comparing both j and chi differ by no more than 2.85e−14. All strict shell, future-energy/conservation and reciprocal literal-current diagnostics remain within their unchanged 1e−10 criteria. Counts are **314 passing checks plus six retained non-admission diagnostic failures**, not 320 passing checks.

Order32 results:

| Radiation and direction | Number R (fm−3 s−1) | Energy Q (MeV fm−3 s−1) |
| --- | ---: | ---: |
| Empty F | 4.605122850209086e−27 | 5.095197948127051e−27 |
| FD(mu=0) F | 2.923698762260435e−27 | 3.234859011051718e−27 |
| FD(mu=0) B | 7.176074613029103e−28 | 7.939802118324058e−28 |

No small net source is reported as an accuracy certificate. FD(mu=0) is deliberately not imposed LTE with the matter-set chemical potential, and no inverse is manufactured from detailed balance.

## Boundaries of admission

The author status supports selected higher-order numerical quadrature of these positive branches on this one interval and bath, subject to DIFFERENT review. It does not qualify order12, all incident energies, the lower beta spectrum, another bath, an EOS solve, gradients, total moment parity or a transport configuration.

The edge tests isolate three inner axes and an alternate radial integration, but do not prove uniform interior inner error or isolate phi. The semi-infinite radial rule and finite tail90 comparison are converged numerical evidence on tested states, not a rigorous uniform omitted-tail bound. These limitations remain explicit. The prior coarse and local-tail failures remain valid evidence against their original quadrature settings.
