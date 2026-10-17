# Independent numerical review: raw finite-T RMF coupling sensitivities

15 October 2026. **Accept the first-order parameter interface at the independently tested smooth warm point.** `probe_parameter_ad.py` executes **84 passing checks**, saved with source hashes and raw finite-difference ladders in `parameter_ad_review.json`. The author's 504-check report and separate physics review remain distinct evidence; they are not claimed as this review's reruns.

The new state is `(nB,T,Yp)=(0.27,17,0.22)`. All five baseline couplings are deliberately shifted from the rounded GM1 values: factors `(1.005,0.997,1.01,1.002,0.998)` multiply `(Cs,Co,Cr,b,c)`, respectively. Bare M remains 939 MeV. This checks genuinely supplied coupling inputs rather than an accidentally hardcoded GM1 Jacobian. These raw perturbations are not a recalibrated EOS family or physical uncertainty distribution.

The direct nonlinear primal residual is gated before derivatives. A separate finite-difference species chemical Hessian, holding the other species density fixed, independently confirms positive eigenvalues and reproduces the supplied local Hessian spectrum. Every coupling-perturbed primal used below also passes its normalized nonlinear residual gate. No global branch or phase certificate follows.

Independent three-step central finite differences re-solve the full nonlinear state at each raw coupling perturbation, with fractional steps 0.002,0.001,0.0005. Richardson extrapolation agrees with every output derivative in all five coupling columns, and the two Richardson levels converge. All twelve callback primals match the direct solver; JIT jacfwd and JIT jacrev match the supplied 12x8 Jacobian. Mixed state/coupling JVP and VJP dot-product tests pass. Nested AD raises the expected unsupported-callback error.

The explicit scalar-equation parameter row has the correct signs and units: `(-Phi/Cs²,0,0,M Phi²,Phi³)`. The implicit state matrix uses the supplied couplings and the same qualified distribution moments. The reduced canonical Hellmann-Feynman terms are `(-Phi²/(2Cs²),nB²/2,n3²/8,M Phi³/3,Phi⁴/4)`, multiplied by HBARC for returned units.

Independent vector-energy identities verify the pressure parameter-work subtraction exactly: derivatives of energy, free energy and pressure with respect to Co equal `HBARC*nB²/2`; the corresponding Cr derivatives equal `HBARC*n3²/8`. Entropy has zero direct vector-coupling derivative. Omitting the canonical work from varying-parameter Gibbs-Duhem would double those pressure derivatives. Both the analytic checks and re-solved finite differences reject that erroneous alternative.

The parameter derivatives hold independently specified nB,T,Yp and all other raw couplings fixed. A calibration-constrained variation, saturation-property parameterization, beta-equilibrium composition response or density-dependent coupling requires its own chain rule and physical model. No bare-mass, omega-rho, higher-order or phase-selection derivatives are qualified here.

Pinned-source explicit partials check NMB conventions; they do not replace the still-missing matched external finite-temperature forward comparison. Callback conditioning, local phase restrictions, NaN/strict-preflight contracts and first-order limitations from `rmf_ad_review.md` continue to apply. No author files or dependencies changed.
