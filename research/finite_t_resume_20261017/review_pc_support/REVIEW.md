# Independent selected soft-PC reference review

Accepted as a selected numerical reference at the frozen free T=2 MeV chemical bath and full tree-level current. The original PR55 q-axis shell failure is not repaired in PR55 by this review. No production source changed. This is a research-leaf alternate integral.

Candidate `conditional_pc.py` SHA256 `5083e7857f3a9e256f4565b6ba19ce87f572b84abf48a381774f2bb364262e2f`; author manifest `3323579161e95b79d378bfdb0d4692bbb0c281dfa12d0b7d6be65202cbaa9c7c`. Candidate-blind preparation checkpoint `14410969978f1694a3bf096e499ea83313b28447e272a37e6c097ccbf4d21ccb` preceded candidate inspection. The actual four-call LOCK preceded every new output. All 28 initially pinned author files and five candidate-blind preparation files remain unchanged. Initial failed q-axis and unmapped-endpoint evidence retain their original hashes.

## Forward results before reciprocity

Exactly four fresh integrals used 4.71 s measured integration time, each in a separate bounded single-thread process: two independent rules at 0.1 MeV, one independent rule at the new 0.13 MeV incident energy, and one candidate rule at 0.13 MeV. There was no new 0.12 MeV integral. Sixty-seven forward, support, representation and provenance checks pass; zero fail. Provenance assertions are included in this count, not additional physical benchmarks.

| Comparison | relative j difference | relative true-chi difference |
|---|---:|---:|
| Independent 0.1 MeV lower/higher rules | 2.274e-12 | 0 |
| Higher independent 0.1 vs frozen author | 8.640e-12 | 5.002e-12 |
| New independent 0.13 vs fresh candidate | 5.457e-12 | 3.638e-12 |

The prelocked separate-branch gate is 3e-6. Differences use `abs(expm1(log_a-log_b))`; small negative differences cannot pass merely by their sign. At 0.13 MeV the independent natural logarithms are log j=-2813.0367774949973 and log chi=-2814.376459579808. Both ordinary values are unrepresentable and remain `None`; no zero support is inferred. At 0.1 MeV the independent higher-rule logs are -3786.5848597671716 and -3787.9395418520025. Both directions use their actual distinct FD products. KMS is checked only after these absolute forward comparisons.

## What is independent

The oracle extracts only the pinned old `relativistic_capture_oracle/events.py` explicit u/v spinors, vertex, spin contraction and Lorentz boost primitives through an owned AST selector. It does not import the candidate scalar contraction or copied PR55 current. The literal inverse-current average is over the initial proton spin and the antineutrino has one active helicity. The candidate instead uses its forward-average-four scalar contraction with the reciprocal factor two.

The oracle integrates the original cap coordinate xi=1-u using Gauss–Jacobi beta=1/2, rather than the candidate's v=sqrt(1-u) Legendre substitution. Its laboratory neutrino lies along z and the proton carries the conditional polar cap; the candidate places the proton on z. The oracle assembles the unreduced target phase density, invariant Møller flux and physical differential cross section separately. It uses independently written direct FD log products.

Both calculations use the same exact two-body CM event family and generalized Laguerre radial weight. These shared ingredients limit independence: this is different numerical realization/current evaluation of the same physical Hamiltonian, not an external physical validation or a completely unrelated phase reduction. Earlier accepted sources remain named dependencies. The independent incoming-angle mapping and unreduced normalization were specified in candidate-blind preparation; the .13 MeV comparison was locked before output.

## Support and normalization

The actual inverse reaction is anti-nu+p -> e+ + n. With K=[(Mn+me)^2-Mp^2]/(2E), the soft branch K>Mp requires proton energy A >= (K^2+Mp^2)/(2K). At E=.1 this is 8498.88313464223 MeV in the precomputed Decimal60 anchor. The stable positive cap uses A+p-K=(A-A0)[1+(A+A0)/(p+p0)]. Neither a cold stationary threshold nor an ordinary underflow value truncates it. The independent minimum agrees within 2e-14 relative. Every sampled final state is future directed; independent shell, four-momentum and original unsquared-delta checks pass stricter prelocked gates 2e-11, 2e-12 and 2e-12. The original far-ultrarelativistic coordinate witness remains an unresolved numerical witness of the original code, not an absent channel.

In natural momentum units, the proton spin-two target phase after azimuth is 2 p A dA dc/(4pi^2 hbar), after the hbar-c cross-section conversion. Multiply by Møller flux (s-Mp^2)/(2 E A) and d-sigma/d-Omega = amplitude_avg2 pstar/[64pi^2 s p_initial]. Using p_initial=(s-Mp^2)/(2sqrt(s)) and amplitude_avg2=2 amplitude_avg4 gives p pstar amplitude_avg4/[64pi^4 E sqrt(s) hbar], exactly the author's unfitted measure. There is no neutrino DOS in a per-occupation coefficient. A later volume moment would introduce the one-helicity E^2/[2pi^2(hbar c)^3] once.

True absorption uses f_p (1-f_n)(1-f_e+); spontaneous emission uses (1-f_p) f_n f_e+, with positron chemical potential -mu_e. No external radiation blocking enters these coefficients. Actual radiation would subsequently give j(1-f_nu)-chi f_nu. Chi/c is the inverse absorption length, while (j+chi)/c is the relaxation inverse length. Finite negative logarithms denote positive coefficients, not negative rates.

## Qualification limits and retained failures

The author separately tested inner refinement, unchanged-interior panel extension tail70 to90, and semi-infinite Laguerre orders8/16/24/32 at .1/.15 MeV. These are distinct controls. The new oracle shares the semi-infinite radial family and does not furnish a rigorous tail remainder bound. Author .15 has not received a fresh independent integrated check in this bounded review. The independent .1 refinement changes multiple orders together; it does not independently isolate every axis error at .13.

The initial square-root incoming-endpoint failure and original q-axis shell failure are retained unchanged; their old sources remain failed at their recorded gates. No positive panel is discarded by weighted-smallness or ordinary-underflow arguments. No uniform subthreshold domain, arbitrarily small-E conditioning, complete energy moments, finite-band closure, transport/evolution, EOS/GM1 coupling, gradients, empirical corrections or model uncertainty is qualified. General reusable API/input qualification is also not established by this selected reference review.

All four launches had fresh exact account/window/reset guards: 19:16:56 UTC 92%, then 19:17:22/26/29 UTC 93%, reset1792652325. The earlier author reset+3s anomaly remains documented in the parent campaign; no identity criterion was relaxed. No further physical calls were made.
