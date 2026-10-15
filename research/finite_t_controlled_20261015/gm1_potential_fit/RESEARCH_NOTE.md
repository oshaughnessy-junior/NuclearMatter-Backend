# Compact one-potential GM1 thermodynamic fit

2026-10-15. Author research checkpoint; independent review required. This is a deterministic60-coefficient analytic fit to the existing exact **internal** GM1 reference, for controlled local bulk-thermodynamic experiments. It is not a learned replacement for the nuclear chain, a production EOS, a weak-response fit, or external physical validation. The matched external finite-temperature GM1 forward gate remains **OPEN**.

## Domain, one potential and units

The evaluation interface enforces nB∈[0.18,0.26] fm⁻³, Yp∈[0.18,0.33], T∈[0,3] MeV. Within that rectangle, measured errors below apply to specified blocked samples, not a uniform error bound. The potential contains nucleons only: no leptons, photons, nuclei or beta-equilibrium constraint. GM1 parameters, source conventions and common bare M=939 MeV are fixed to the frozen reference; coupling derivatives are not fitted.

Use free energy per baryon

    a(n,T,Y)=M+C0(n,z)+T² C2(n,z)+T⁴ C4(n,z),  z=(1−2Y)²,
    Ck=sum_{i=0..4,j=0..3} c[k,i,j] Cheb_i(x(n)) Cheb_j(v(z)).

x and v map the declared n and z intervals to[−1,1]. The isospin-even basis implements the symmetry of the common-mass GM1 neutron/proton ansatz. It does not assert an API domain at the reflected, untested compositions. C0 has MeV units, C2 MeV⁻¹ and C4 MeV⁻³. There are20 coefficients in each of three fields, stored in `coefficients.json`; the scalar field need not be solved when evaluating the fitted potential.

Every returned thermodynamic quantity comes from this **single a**:

    f=n a,
    s=−n a_T,
    e=n(a−T a_T),
    P=n² a_n,
    mu_n=a+n a_n−Y a_Y,
    mu_p=a+n a_n+(1−Y)a_Y,
    Cv|n,Y=−n T a_TT.

Energy/free-energy/pressure densities are MeV fm⁻³, entropy/Cv are fm⁻³ with kB=1, chemical potentials MeV. Analytic Chebyshev derivatives include the full z(Y) chain rule. In particular z_Y=−4(1−2Y) and z_YY=8; scaled-variable factors are included. Thermal derivatives vanish appropriately at T=0. The chemical Hessian uses independent nn,np, not only fixed-Y compression:

    Hnn=2a_n+n a_nn−2Y a_nY+Y² a_YY/n,
    Hnp=2a_n+n a_nn+(1−2Y)a_nY−Y(1−Y)a_YY/n,
    Hpp=2a_n+n a_nn+2(1−Y)a_nY+(1−Y)² a_YY/n.

These expressions retain a symmetric full two-charge Hessian in MeV fm³. Euler/Maxwell thermodynamic identities are consequences of this representation; their satisfaction alone is not evidence of accuracy against GM1.

## Physical baseline and data generation

Use five training densities[0.18,0.195,0.21,0.225,0.24] and four compositions[0.18,0.22,0.26,0.30]. At each of20(n,Y) points, evaluate the frozen cold branch and full finite-T state at T=2 MeV:40 actual EOS states. Their field/density constraint residuals are gated before fitting. Sources are unchanged `rmf_reference/gm1_finite_t.py` and `rmf_phase/stability.py`; exact hashes and source-snapshot provenance accompany the checkpoint.

C0 is fitted to cold a−M. C2 starts from the source's cold density-of-states/Sommerfeld coefficient,

    gamma=sum_i kFi sqrt(kFi²+mD²)/(3 hbarc),  C2=−gamma/(2n),

using the cold self-consistent Dirac mass in internal fm units. C4 is fitted from the remaining full T=2 free energy. Thus C4 is an effective correction at that sampled temperature and can absorb higher powers; it is not a claim to have derived the exact fourth-order asymptotic coefficient. The final fit is compared to the **full** exact reference at held-out temperatures. The much narrower qualified domain of the old `sommerfeld_thermal` convenience API is not silently extended by this fit.

Each coefficient field uses a20×20 Chebyshev system, condition number143.9; no regularization, neural network or rate-data calibration is used. M is subtracted before fitting the cold field. Reported free/energy errors are absolute per-baryon errors, rather than small relative errors dominated by the939 MeV rest term.

## Model selection and honest blocked tests

The initial direct-Y polynomial fit is preserved completely under `initial_Y_polynomial/`, including source, coefficients, data and results. It passed500 original thermodynamic/stability gates, but its chemical-difference error reached0.0630 MeV and a sampled balance-factor discrepancy8.1%. Those diagnostics informed the isospin-even basis choice. The initial holdouts therefore cannot be presented as untouched final tests.

The final basis and acceptance gates were fixed before evaluating a **new**32-point blocked set. No random neighboring split is used:

- density block: n=[0.248,0.258], Y=[0.19,0.27], T=[0.8,1.4];
- composition block: Y=[0.315,0.325], n=[0.188,0.228], T=[0.8,1.4];
- temperature block: T=[2.4,2.9], n=[0.188,0.228], Y=[0.19,0.27];
- joint block: n=[0.248,0.258], Y=[0.315,0.325], T=[2.4,2.9].

Each designated blocked coordinate lies beyond the corresponding training hull. This tests modest extrapolation, but does not establish arbitrary extrapolation or a continuous-domain certificate. New frozen coefficients should face further independently chosen checks before integration elsewhere.

## Errors and stability

Across the final blocked points:

| Quantity | Largest error |
|---|---:|
| free energy per baryon |0.0002473 MeV|
| energy per baryon |0.0002891 MeV|
| pressure |0.01763% relative|
| entropy density |0.02323% relative|
| Cv |0.06363% relative|
| mu_n |0.008830 MeV absolute|
| mu_p |0.009196 MeV absolute|
| mu_n−mu_p |0.003381 MeV absolute|
| full chemical Hessian |0.1215% Frobenius relative|
| exp[(mu_n−mu_p)/T] factor |0.3710% relative|

The last row is an affinity-sensitivity diagnostic only. It is **not** weak-rate accuracy: dispersion, response and matrix elements are not determined by this bulk fit, and mixing fitted chemical potentials with an unchanged microscopic kernel can break consistency. No microscopic Dirac/Landau masses, fields or currents are inferred from the polynomial.

The full exact and fitted chemical Hessians are positive at every blocked point, with minimum eigenvalues710.02 and709.95 MeV fm³ respectively. Exact canonical scalar curvature is also positive at those states, minimum0.12938 fm⁻². Fitted Cv is positive. A separate567-node **fit-only** rectangle scan has minimum chemical eigenvalue648.88 MeV fm³ and positive Cv/T; it does not verify exact EOS error between sampled points or global phase coexistence. The fit is deliberately restricted away from the spinodal examples. No binodal or phase construction is supplied.

The full-reference Cv comparison is obtained from entropy differences at two decreasing T steps, with a separate convergence gate. Another18 checks cover two blocked cold states and T=0.05 entropy/DOS limits. An additional33 analytic-implementation checks independently finite-difference the fitted potential to verify its chemical potentials, entropy, Cv and full Hessian, test the intrinsic symmetry and enforce the domain guard. Automatic identities/stability scans are distinguished from reference-accuracy evidence.

## Failed fit, cost and reproducibility

A genuine12-coefficient low-order baseline is preserved as `baseline_coefficients.json` and `baseline_results.json`. On the final blocked set it fails the same bulk gates: maximum free error1.24 MeV per baryon, pressure error39.8%, Hessian error61.0%. Thermodynamic consistency by construction did not save that inaccurate model. The initial direct-Y low-order failure and accepted-but-less-accurate first model are preserved separately; no failed fit is relabeled as accepted.

The first physical experiment cost46.92 s. The final physical experiment cost44.88 s, plus less than1 s to record canonical scalar curvature from already-solved states;18 cold/low-T checks cost0.37 s and33 analytic checks0.20 s. Partial training/holdout checkpoints are written throughout, with a105 s experiment guard. No physical grid was interrupted or numerically failed in these runs. The failed low-order model is a model-accuracy failure, not an EOS solver failure.

Final author evidence comprises596 main/canonical-curvature checks,33 analytic checks and18 cold/low-T checks, all passing. Run `fit_study.py`, `analytic_checks.py` and `cold_limit_checks.py` in this directory; the supplemental `record_scalar_curvature.py` records the same canonical-curvature checks from an existing results file without repeating forward roots. All source/data/hash manifests are retained. No production file, dependency, upstream source, git state or paper repository was changed by this leaf.

The intended next use is a small, explicitly qualified bulk-potential component in controlled EOS experiments. Production promotion, coupling-parameter sensitivity, a general finite-T fit library, direct-Urca fits, merger/cooling phenomenology and external GM1 validation require separate work. Independent review of the frozen fit and fresh blocked states comes next.
