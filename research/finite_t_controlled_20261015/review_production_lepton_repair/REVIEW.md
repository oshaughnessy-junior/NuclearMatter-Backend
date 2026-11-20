# Separate independent recheck of scaled lepton quadrature repair

**Selected numerical repair ACCEPTED; candidate integration remains BLOCKED by a concrete minimum-version API incompatibility.** There are665 passing numerical/API/provenance checks and one static compatibility failure. Original independent review207 PASS/4 FORWARD FAIL is unchanged. No candidate or upstream file was modified by this reviewer.

Reviewed module SHA256`e562135234bcbd9c27e56e5b622253d5a3d2420c52c0b3fc67f5f383918e5a76`, candidate manifest`599bc681945ff344d987f77a557565337be44ce17509fabbca2d36bd1c47de7c`. Exactly the five previously scoped paths remain proposed. The five staged files and current author report/checkpoint/manifest are copied into `reviewed_repair_candidate/`; all original candidate/failed-review snapshots remain untouched.

## Remaining integration blocker

The repaired module unconditionally imports `scipy.special.log_expit`. The package still declares `scipy>=1.7`; the [official SciPy function documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_expit.html) explicitly states that log_expit was added in1.8.0. Therefore the current module cannot import under the declared SciPy1.7 API. This is a definite static required-API mismatch, not merely an unexecuted compatibility test.

`compatibility_probe.py` supplements that primary evidence by temporarily removing only this export from the existing SciPy module in an isolated review process, then importing the candidate under a fresh legitimate package name. Import raises `cannot import name 'log_expit' from 'scipy.special'`. The export is restored in finally. This controlled missing-export simulation is **not execution of SciPy1.7**, an installed minimum-version environment or an altered/fake package initializer. No dependency bump or source repair was made. A separate author repair and recheck must preserve this failed record.

## Original forward failures replayed and repaired

The original module is imported under a unique name inside the real package namespace directly from the preserved snapshot. All four original failures reproduce; their old errors are retained. The new calculation is compared at the SAME unmodified3e-8 density gate:

| Electron m/T, mu=0 | Preserved original relative error | Repaired relative error |
|---|---:|---:|
|40|1.6408275659e-4|-3.66e-15|
|100|8.8569836169e-4|8.88e-16|
|300|8.9687168811e-4|-4.33e-15|
|600|8.9969860842e-4|7.11e-14|

These populations remain positive and normally represented, not underflow states. Independent all-field checks also pass at each original state and NEW signed heldouts mu/T=+/-.53 at the same four m/T values. My oracle uses the absolute scaled-Bessel particle density and original-momentum Gauss integration of the centered kinetic-energy variance. Its Cv expression is a sum of positive terms, avoiding the tiny-squared-moment problem in the author's first oracle. New dilute all-field discrepancies are at most9.86e-14. Independent96/144-node moment refinement and separate source tail60->90/rtol2e-12->4e-13 pass without relaxing gates.

The author's retained two covariance-oracle underflow failures and ten insufficient-order momentum failures remain historical failures. They were not overwritten, reclassified as successes or used to change this review's gates.

## Audit of the repaired thermodynamic chain

The source changes preserve the same ideal spin-two particle/antiparticle FD integrals. x=(E-m)/T removes dimensionful quadrature floors; separate population occupation scales and zero epsabs prevent a tiny antiparticle from disappearing merely because a larger component dominates an absolute error target. Log restoration combines occupation and dimensional scales before ordinary rounding, rather than imposing a positive density floor. Entropy's scaled expression is the same occupation entropy; the stable small-mu response retains the difference of occupation variances before cancellation. The dimensions of restored density, energy, entropy, susceptibility, fixed-net Cv and mu_T remain correct.

The response identities remain chi=A0/T, n_T=A1/T², mu_T=-A1/(T A0), Cv=(A2-A1²/A0)/T². Sign reversal changes net number and the appropriate responses while exchanging particles/antiparticles. f=mu*n_net-P is the free-energy identity, now avoiding a rest-dominated e-Ts subtraction. The thermal increment is the sum of positive holes, excitations and antiparticle terms relative to the same cold net density. No extra rest-mass/Q term enters.

Each flavor still has independent total mu or signed net NUMBER, with opposite antiparticle mu. No implicit electron/muon equilibrium or charge neutrality is introduced. Energy contains both populations' rest masses; Cv holds net number fixed while pairs equilibrate. This remains a CPU reference, not JAX/cold/cooling replacement.

## New independent heldouts and package checks

`probe.py` gives594 PASS, zero numerical failures. New massive states (T,mu,m)=(.63,1.19,M_E),(4.73,-.83,M_E),(18.23,-98.7,M_MU),(.217,147.3,M_MU) are compared against an independent original-momentum rapidity oracle at72/112-node rules, including every thermodynamic/response field and free energy. Maximum discrepancy is1.65e-11. The prior oracle FUNCTION is extracted without importing or running its frozen driver, so no old review outputs are rewritten. Warm thermal increments also agree with an independently reconstructed rest-inclusive cold energy subtraction.

New exact massless cases T=2e-41,7e-7,3e7 MeV with mu/T=0,+/-.61 check every field, separate particle/antiparticle scalar momentum integrals and analytic thermal subtraction. Maximum discrepancy is3.85e-14. The dimensionless analytic formulas avoid underflow in the oracle itself; all tested expected values are ordinarily represented. No scale-dependent absolute assertion floor is used.

Fresh fixed-net Cv and mu_T finite differences follow independent unperturbed warm forward checks. New massive signed roots +/-4.7e-29 fm^-3 reconstruct and agree with pair susceptibility. New muon Sommerfeld Cv/stable thermal increment checks pass at T=.0009,mu173.2 MeV. An actual current cold-package JIT call at n=.0173 fm^-3 matches electron+muon number/energy after HBARC conversion; the actual cold energy Hessian is finite and positive. This tests unchanged cold contracts without implying that SciPy quadrature becomes differentiable.

Unresolved susceptibility still raises ArithmeticError, while an ordinary underflowed antiparticle population can coexist with positive resolved susceptibility. Finite-real/non-Boolean malformed scalar attacks reject. Python3.9 grammar parsing passes, but does not imply minimum runtime compatibility. The detached-state/unused-field authenticity limitation from the original review is unchanged; this utility is not an untrusted-code defense.

An independent Git-blob reconstruction in `compatibility_probe.py` gives71 additional passing baseline-package checks. The real24 package imports are preserved; an additional unique snapshot-module import exists only for the old-failure replay. Candidate source bytes match the five-file manifest. The original20-file independent review pin set and the complete author tree are verified unchanged before/after. Current source/evidence preservation and execution identities remain distinct.

## Remaining release limits and reproduction

Executed existing environment is Python3.12.12, NumPy2.3.5, SciPy1.15.2, JAX0.10.1. No installs, production checkout changes, remote PR, paper or git operation occurred. Exact commands:

```
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/review_production_lepton_repair/probe.py
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/review_production_lepton_repair/compatibility_probe.py
```

Both are bounded below120s. Numerical acceptance is selected-state evidence, not a uniform extreme-domain/relative-subnormal certificate, empirical uncertainty, public log-density API, complete matter EOS or derivative qualification. Normal admitted repository CI/full suite, supported/minimum-version execution, conda/pixi NumPy<2.3 and packaging remain open even after the static import blocker is repaired. A scoped DRAFT PR can be prepared after that repair/recheck; this candidate must not be integrated or merged on the basis of the present numerical passes alone.
