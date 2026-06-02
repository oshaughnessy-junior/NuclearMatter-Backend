# Dev log — 2026-06-02 (f) — thermodynamically consistent interpolation

**Who:** R. O'Shaughnessy (with AI pair).  Roadmap Phase-1 action #2 (interpolation).

## Headline (RO, measured against RePrimAnd 1.7)
With the surface-convention fix the production solver now matches **RePrimAnd to
~3–10 m in R** (was ~200 m) and **<0.1% in Λ** — at the ≲10 m budget.

## What changed
`eos.base.at_h` replaced the independent log-linear interpolation of (p, ε, cs²)
with a **thermodynamically consistent** scheme:
- Cubic-Hermite of p(h) with node slopes = exact dp/dh = ε+p ⇒ ε = dp/dh − p is
  *derived*, and dp/dh = ε+p holds by construction (consistent structure sector).
- cs² interpolated from accurate node values: consistent at nodes, **free to be
  non-monotone** (never monotonized; it dips to ~3e-5 at a transition — the hybrid
  solver handles it). We explicitly do NOT take cs² from d²p/dh² (a 2nd derivative
  of a C¹ interpolant is far too noisy — it gave a 360× wrong gradient when tried).
- Parameter is h ≡ ln(μ/μ_surface); chemical potential μ = (ε+p)/n exposed as
  `EOSTable.mu`, so realistic tabulated EOS can be driven by μ or h.

## Why (RO's points)
"Thermodynamically consistent interpolation is critical; realistic systems use
chemical potential / enthalpy as the parameter; cs² isn't monotone at all."  All
three are now respected: consistency via the exact slope identity; μ/h as the
parameter; cs² derived/interpolated, never monotonized.

## Results
- Λ *value* accuracy improved ~25× (M=1.31: 0.109% → 0.004% vs SciPy); R values 1–3 m.
- Non-monotone-cs² hybrid (quark core) matches its (H,β) reference to 1.7 m / 0.004%.

## Honest gradient finding (characterized, not a bug)
The (physically real) thin, stiff low-cs² surface shell makes forward-mode
*tangents* tolerance-sensitive:
- Λ-gradient **converges to the FD truth** as rtol tightens (1e-8 ~90% → 1e-10 ~16%
  → 1e-12 ~3%): a tangent-resolution effect, since the adaptive controller is sized
  for the primal.
- R-gradient has a ~0.5% interpolation-tangent residual (the hand-rolled Hermite is
  less smooth under AD than `jnp.interp`).
Guidance: **default tolerance is fine for values; use rtol ≲ 1e-10 for gradient /
Fisher work.** Future: a smoother differentiable interpolation primitive to tighten
gradients without the value/consistency cost.  Tests updated to assert the
convergence-with-tolerance behaviour (R, Λ, M_max gradients). 27 tests pass.

## Next
Attribute the last ≲10 m of the RePrimAnd R residual; then move to Phase 2 (RMFT)
or harden the gradient interpolation, per priorities.
