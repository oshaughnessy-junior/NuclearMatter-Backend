# Backlog / deferred development items

Tracked items intentionally deferred, with the context for when they matter.

## Gradient tightening (forward-mode tangent precision)

**What:** the thermodynamically consistent Hermite interpolation (`eos.base.at_h`)
gives excellent EOS/TOV *values* but the forward-mode AD *tangents* are
tolerance-sensitive and have a small interpolation-tangent residual:
- Λ-gradient converges to the FD truth only as rtol tightens (1e-8 ~90% off →
  1e-10 ~16% → 1e-12 ~3%), because the stiff low-cs² surface shell under-resolves
  the tidal tangent under the primal-sized adaptive controller.
- R-gradient has a ~0.5% residual (the hand-rolled cubic Hermite is less smooth
  under AD than `jnp.interp`).

**Why deferred:** values are at the ≲10 m / <0.1% budget now, and we don't yet need
high-precision gradients. We *will* need them for **(a) Fisher-matrix error
estimates** and **(b) NUTS-like (HMC) sampling** of EOS parameters — both consume
dΛ/dθ, dR/dθ, dM_max/dθ and are sensitive to gradient accuracy/smoothness.

**Likely fixes (when we get there):**
1. A smoother differentiable interpolation primitive (monotone PCHIP via a
   JAX-clean implementation, or expressing the Hermite through `jnp` primitives that
   differentiate cleanly across moving node boundaries).
2. Carry `p` as an integrated ODE state with dp/dh = n·μ (n log-linear, clean
   gradients) so dp/dh = ε+p is exact *and* the tangent stays smooth.
3. Forward-over-reverse or a tangent-aware step controller for the tidal sector; or
   simply document/standardize rtol ≲ 1e-10 for gradient/Fisher runs.

**Acceptance:** dΛ/dθ and dR/dθ within ~0.1% of FD at default tolerance, smooth
enough for HMC step-size adaptation.
