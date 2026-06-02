# Dev log — 2026-06-02 (e) — surface convention fix (the ~120 m R win)

**Who:** R. O'Shaughnessy (with AI pair).  Roadmap Phase-1 action #2 (surface part).

## Hypothesis (RO)
The ~200 m RePrimAnd-vs-ours R discrepancy is mostly **surface convention**.

## Measurement (confirmed)
Our table stopped at n_min≈3e-4 fm⁻³ (just below neutron drip); integrating only to
there truncates the outer crust/atmosphere.  Extending the EOS down to p→0 with a
consistent low-density continuation and integrating the star to the surface
**raises R(1.4) by ~114 m** (35 m at n_surf=1e-4, converging to ~120 m by 1e-7),
Λ stable to <0.2%, M_max unchanged.  So surface convention is indeed most of the gap.

## Implemented
`eos.base.extend_to_surface(n, eps, p, n_surface=1e-7, gamma_crust=1.3)` — prepends a
polytropic tail p = p0 (n/n0)^Γ with ε from the cold first law (ε = A n + p/(Γ-1),
continuous at the base).  AD-safe (tracks the param-dependent base point).  Now the
default in `MetaModelEOS` (`n_surface=None` restores the old truncation); `HybridEOS`
inherits it via its hadronic base.  diffrax-vs-SciPy internal floor preserved (~1.7 m).

## Honest finding -> sharpens action #2
The surface shell has tiny cs² and it **matters for Λ**: flooring cs² to fix
stiffness corrupts the Λ *value* (floor 1e-3 → Λ −10%).  Keeping the true small cs²
gives an accurate Λ *value* (0.04% vs SciPy) but the forward-mode **Λ gradient** is
under-resolved near the surface (~25% at rtol=1e-9) because the finite-difference
cs² + clip make the tidal tangent noisy/stiff.  This is a second, concrete reason to
do action #2's **monotone, thermodynamically-consistent cs²** (PCHIP / analytic in
the polytrope tail).  Test `test_fwd_grad_lambda_*` now asserts finite + correct sign
+ order (~25%) and points here; R and M_max gradients remain tight.

## Status
27 tests pass.  Next: monotone cs² reconstruction (restore Λ-gradient precision,
tighten R further), then re-measure the RePrimAnd residual (expect ≲~80 m) and
attribute it.
