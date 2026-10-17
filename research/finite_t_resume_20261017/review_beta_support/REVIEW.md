# DIFFERENT selected upper-beta spectrum review

Accepted as a selected alternate numerical reference at the fixed T=2 MeV free-particle chemical bath, Eanti=1.2 MeV and the new independent incident point1.27 MeV. This does not qualify the original PR55 tail rule, the earlier [1.1,1.4] incident panel or a whole-energy source. No production or candidate bytes changed.

Candidate `conditional_beta.py` SHA148c99c37dc0550ca983b53d86123e25125469b15fd2fde9edc16c169c2ac191; author manifest acec26c884203ed396e7d9ff0684a31568f172826a692b50a52fb589380547f6. Candidate-blind preparation checkpoint6257b765dbe024b655ed3b9980470187f3c26f82ad125d8f320cafd5dba07dcc preceded all candidate inspection. The review's actual-map/profile LOCK was frozen before all four fresh outputs.

## Separate positive forward comparisons

Exactly four fresh integrals completed in1.025s measured integration time, each in a bounded isolated single-thread process. These are the prelocked two independent1.2 rules, independent1.27 and candidate1.27. No additional axis, tail, incident-band or whole-energy calls were made. Seventy-four checks pass with zero new failures; this includes source/preservation assertions and is not a count of distinct physical models.

| Separate comparison | relative j error | relative true-chi error |
|---|---:|---:|
| Independent1.2 lower16/higher24 coupled profiles | 7.106e-15 | 2.132e-14 |
| Higher independent1.2 vs frozen candidate author | 8.953e-13 | 8.953e-13 |
| New independent1.27 vs fresh candidate | 8.953e-13 | 9.095e-13 |

All use the unchanged3e-6 gate and absolute `expm1(loga-logb)` differences. At1.27, independent log j=-66.4884001654086, log chi=-67.25808225021851, with ordinary j=1.3318481064494823e-29 and chi=6.168591560178284e-30 s^-1. These are positive representable coefficients. Independent future-shell, full-vector conservation and original unsquared-delta checks pass2e-11/2e-12/2e-12. Reciprocal literal amplitudes and posterior KMS also pass, after absolute branch comparison.

## Support and numerical independence

For parent neutron energy A and momentum p, fixed laboratory anti-neutrino E gives the proton-electron invariant s=Mn^2-2E(A-pc). Its threshold s0=(Mp+me)^2 implies D=[Mn^2-s0]/(2E). For E above the neutron-rest endpoint E0, the actual parent support is A >= (Mn^2+D^2)/(2D), not A>=Mn with a stationary emitted-energy cutoff. The candidate-blind Decimal60 minima are1027.031832217423 MeV at1.2 and1052.209458794066 MeV at1.27. Independent double roots agree within2e-14 relative.

The oracle uses rest neutrino energy and rest electron Dalitz energy, integrating the isotropic rest-direction delta with its exact1/(2gamma v e*) Jacobian. Its rest-neutrino endpoint gap uses a Gauss–Jacobi rule, electron energy its exact affine interval, and relative orientation a periodic azimuth. The rest electron longitudinal momentum is reconstructed from the original neutron/proton shell relation, rather than reusing the candidate pair-CM outgoing-angle event map. Laboratory vectors are rotated and boosted explicitly; actual separate FD products use their laboratory energies.

Both methods use threshold-shifted semi-infinite parent-energy generalized Laguerre alpha3/2. That shared radial representation and the same physical Hamiltonian limit independence. The candidate and oracle also evaluate the same underlying literal-spinor contraction algebra: selecting an older independently pinned source namespace is not a new independent current test. Independence here is the rest-Dalitz/cone-delta phase reduction, coordinate sampling, factorized absolute normalization and independent occupation assembly. It is numerical agreement within a specified tree-level model, not external model validation or empirical uncertainty.

## Absolute normalization and actual branch products

The old literal amplitude is neutron-spin-averaged-two, GF^2 Vud^2/4 times its full spin sum. Use the original rest Dalitz measure amplitude/(64pi^3 Mn hbar) de* dEe*, parent spin-two phase p^2dp/[pi^2(hbar c)^3], proper-time Mn/A, conditional isotropic cone1/(2gamma v e*) and remaining azimuth average. Divide by one active-helicity DOS E^2/[2pi^2(hbar c)^3] once. This reconstructs the spectral per-occupation normalization independently of the candidate's simplified coefficient. An equivalent covariant check gives p pstar amplitude_avg2/[128pi^4 E sqrt(s) hbar] dA dc dOmega from parent spin density,1/(2A), fixed-k1/(2E) and pair phase space. No fitted factor or extra neutrino spin appears.

Direct j uses fn(A)(1-fp(Ep))(1-fe(Ee)); inverse chi uses(1-fn(A))fp(Ep)fe(Ee). Neither direction is manufactured by KMS. Total chemical potentials are rest-inclusive; the fixed bath is T2, mun919.8287578100662,mup916.8449303084382,mue.17446333200826397 MeV, with physical masses Mn939.56542052,Mp938.27208816,me.51099895069 MeV. The full n->p vertex uses qGamma=pp-pn; future emitted lepton transfer is its negative. The genuine inverse is3->1, not a PC substitution. External anti-neutrino blocking is absent from j/chi and would enter later as j(1-fanti)-chi fanti. True absorption length uses chi/c; relaxation uses(j+chi)/c.

## Retained failures and boundaries

Both freshly reproduced original tail70->90 branches still change2.2377617918348388e-4 against3e-6. They remain failed. Their changing finite Gauss interval alters interior sampling; the discrepancy is not a separately measured physical missing-tail correction. The old tail70 value happens to be within~2.19e-6 of the new reference, which does not waive the failed stability gate. All12 prior beta local/panel failures remain explicitly retained, including the ten upper incident-panel positive F/B R/Q failures. They were not rerun or repaired here.

The author separately compared fixed excess-energy panels and semi-infinite parent rules at1.2/1.3; independent new coverage is1.2/1.27 only. Coupled16-to24 refinement does not isolate every angular axis. Shared semi-infinite radial convergence is not a rigorous remainder theorem. General API adversarial qualification, uniform E0<E<=1.4 accuracy, below-endpoint branch switching, upper incident-panel convergence, whole-energy/tail closure, source inventory, transport/evolution, EOS/GM1 binding, AD, native runtime CI and empirical corrections remain OPEN.

All28 pinned author files, five blind-preparation files and13 prior source pins were verified unchanged at completion. Each physical launch had a fresh exact account/window/reset guard at19:43:14/16/17/19 UTC,94% weekly used, reset1792652325. No further physical calls are planned.
