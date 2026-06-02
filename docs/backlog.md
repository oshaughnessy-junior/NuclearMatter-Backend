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

## RMF continuation solver (for strongly nonlinear sets, e.g. NL3)

**What:** the self-consistent scalar gap equation `Phi/C_sigma + bM Phi^2 + cPhi^3 =
rho_s(M-Phi)` is convex and easily Newton-solved for well-behaved sets (GM1, b>0),
but for sets with `b, c < 0` (NL3 and many modern RMF parameterizations) it is
**non-convex** — `resid < 0` at both ends of the physical m* bracket, with multiple
roots. The physical branch must be tracked by **continuation**: solve at low density
(m* ≈ M) and step up in density using the previous solution as the initial guess.

**Why deferred:** GM1 (the default) is fully validated across all six saturation
properties (n0, B/A, K, m*/M, E_sym, L) plus M_max/R, so the *engine* is proven; the
continuation solver is needed only to unlock NL3-class sets.

**Fix:** build the EOS table by a `jax.lax.scan` over the (increasing) density grid
that carries Phi as the running initial guess (a few damped-Newton steps per
density), rather than the current per-density independent solve. This is naturally
robust and stays differentiable. Then validate NL3 (n0=0.148, B/A=-16.3, K=272,
m*/M=0.595, E_sym=37.4, L=118; M_max~2.77, R(1.4)~14.6).

## Muons in beta-equilibrium

**What:** beta-equilibrium NS matter contains muons once mu_e > m_mu (~106 MeV,
around n ~ 0.1-0.2 fm^-3). Currently only electrons are included (RMFT and
metamodel). Add muons via the common lepton chemical potential mu_e = mu_mu with
charge neutrality n_p = n_e + n_mu (an inner solve for mu_e given the charge
density). Small but real effect on the symmetry-energy-driven part of the EOS and on
Lambda/R at the ~0.1-1% level.

