# A single GM1 potential targeted to low-temperature affinity

This NEW internal-reference approximation addresses the previously preserved low-temperature chemical-affinity failure. It does not replace the detailed physical chain. The frozen old fit and its independent review are unchanged. This is common-939-MeV nuclear-only GM1, with no leptons, physical neutron-proton mass splitting, external finite-temperature benchmark, parameter recalibration, or transport kernel.

The explicit evaluation domain is nB=0.18–0.26 fm^-3, Yp=0.18–0.33, T=0.12–3 MeV. The lower temperature bound is deliberate: a finite nonzero chemical-potential fit error cannot uniformly bound exp(delta_mu/T) as T approaches zero. Passing sampled states does not certify a rectangle, every local stability eigenvalue, or every weak rate.

## Diagnosis and construction

Before fitting, development.py reproduced the frozen old independent point n=.1811, T=.12, Y=.1817: delta(mu_n-mu_p)=-.001859757775 MeV, giving a 1.537850575% balance-factor error against the unchanged 1% gate. Original sources/results remain unchanged. That point is explicitly training/development here, never a new holdout.

One analytic Helmholtz free energy per baryon is fitted:

    a(n,T,Y)=939+C0(n,delta²)+T² C2(n,delta²)+T⁴ C4(n,delta²)+T⁶ C6(n,delta²), delta=1-2Y.

Each C is a degree-(6,6) scaled Chebyshev expansion: 196 coefficients altogether. All pressure, entropy, internal energy, chemical potentials, Cv and the full two-charge chemical Hessian are derivatives of THIS ONE potential. There is no separately patched chemical potential or fitted balance factor.

At 65 declared density/composition training points, actual cold, T=2 and T=3 GM1 solves supply 195 physical EOS states. C0 uses a weighted linear least squares system combining free energy, exact a_Y=mu_p-mu_n and a_n=P/n². These derivative constraints improve low-T affinity without violating potential consistency. C2 uses the exact leading cold Sommerfeld coefficient -gamma/(2n); the two warm anchors determine C4,C6 at each training point. The latter three fields then use ordinary Chebyshev least squares. C0 design condition number is 288.20; the other design condition is 193.22. This measured conditioning is not a physical uncertainty estimate. Warm higher-power terms are a restricted approximation, not an asymptotic claim over arbitrary T.

PRE_HOLDOUT_FREEZE.json pinned source, coefficients, basis/domain and gates before any new holdout EOS evaluation. HOLDOUT_PROTOCOL.json then declared 30 new blocked states. No coefficient or basis change followed the holdouts. Density and composition blocks extend beyond the training hull inside the declared domain; additional temperature, joint and endpoint blocks are retained individually. The upper boundaries are therefore sampled extrapolation tests rather than training interpolation everywhere.

## Results and gates

All 198 development checks, 489 blocked-holdout checks and 24 value-only recurrence/complex-step structural checks pass (711 checks total). Development took 33.50s; actual heldout EOS experiment took 34.56s. The repaired OLD point has fractional balance error 1.86084e-6, strictly a development result.

For the 30 new holdout states, maximum errors are:

| Quantity | Measured maximum | Unchanged declared gate |
|---|---:|---:|
| abs(exp(delta(mu_n-mu_p)/T)-1) | 1.3236906e-5 (0.001324%) | .01 (1%) |
| abs(delta(mu_n-mu_p)) | 2.5239640e-5 MeV | balance gate above |
| Free energy per baryon | 1.1509610e-6 MeV | .005 MeV |
| Energy per baryon | 4.4451826e-7 MeV | .005 MeV |
| Pressure relative | 1.4347399e-6 | .01 |
| Entropy relative | 2.9350527e-6 | .01 |
| Cv relative | 1.1460605e-5 | .01 |
| Full Hessian Frobenius relative | 2.4365977e-5 | .02 |

Exact forward density/field residuals are checked before derivative comparisons; perturbed states are also gated. Reference Cv uses two decreasing central entropy-difference steps, with its own refinement gate. Full exact and fitted two-charge Hessians and the canonical scalar curvature are positive at every sampled state. Structural checks use a separately implemented value-only Chebyshev recurrence and complex-step chemical potentials, pressure and entropy, plus density finite differences of those complex-step derivatives. These identities verify implementation consistency; they do not supply independent physical EOS validation.

No new candidate failed gates; the original old-fit failure is reproduced in old_failure_reproduced.json and preserved in its original review. Coefficients/basis were never adjusted to these blocked results. More adversarial holdouts, uncertainty-aware application tolerances and independent review are still needed before broader use. A balance-factor comparison fixes the same lepton contribution and temperature in both models: it does not validate a charged-current prefactor, quasiparticle masses, physical thresholds or a total emissivity. Even an accurate chemical difference does not supply a complete quasiparticle EOS for weak kernels.

## Reproduction and provenance

Run `python holdouts.py` and `python check_structure.py` from this leaf (or by absolute script path). The numerical reference is frozen rmf_reference/gm1_finite_t.py with rmf_phase/stability.py for implicit responses; their source provenance and known non-global phase limitations remain applicable. `development.py` intentionally rebuilds the same training fit and should not be rerun as part of independent holdout review. PRE_HOLDOUT_FREEZE.json and CHECKPOINT.json pin the current files. Independent reviewers should use fresh coordinates without changing coefficients or tolerances.

The matched external finite-T GM1 validation gate remains OPEN. This compact internal approximation supports comparison and controlled sensitivity studies alongside the detailed solver; it is not a surrogate replacement for the physical N2N or weak-rate chain.
