# Backlog / deferred development items

Tracked items intentionally deferred, with the context for when they matter.

## Gradient tightening (forward-mode tangent precision)

> **RESOLVED (via IFT; validated against a re-rooted fixed-mass FD).** The catastrophic
> case was never the solver tangent -- it was differentiating through the *family
> interpolation* at fixed mass. The dominant term is the central-enthalpy
> reparameterization as the mass target shifts; the coarse interpolation got it wrong
> (flipping the sign of dLambda/dcs2_tail). `nmbackend.inference.fisher.sensitivity_matrix`
> now uses the **implicit-function theorem** for fixed-mass observables and the
> **envelope theorem** for M_max, from accurate single-star tangents only.
>
> Crucially, the "13%/137%" residuals we first saw were *artifacts of the noisy
> family-interpolation FD reference*, not the method. Against the correct reference --
> a fixed-mass FD that **re-roots h_c (Newton-polished) for each perturbed EOS** -- the
> IFT sensitivities are: **dLambda/dcs2_tail 0.0%, dLambda/dL_sym 2.5%, dR ~0.3%,
> dM_max ~0.1%** (tests/test_fisher.py). The only residual is the ~2.5% single-star
> Lambda tidal tangent at tight tolerance (the documented item below), which is fine
> for Fisher/HMC. **The n-mu interpolation rewrite is therefore NOT needed** for this;
> keep it only as an optional future smoothing of the last ~2.5%.



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

## Additional RMF parameter sets (verified couplings; metastable c<0 potentials)

**Done:** the scalar gap solve is now a robust, vectorized **first-root** bracketed
bisection (`_solve_phi`): it locates the lowest-Phi (physical) root — the one
continuously connected from m*=M at zero density — for *any* set, replacing the
fragile Newton. GM1 is exact through it across all six saturation properties + M_max.

**Open:** adding a *new* published set is gated by **coupling-normalization
verification**, not the solver. A naive conversion of NL3's (g_sigma, g2, g3) to our
(g/m)^2 + b,c convention failed: the gap equation had **no root** at NL3's known
physical point (m*/M=0.595 @ n0=0.148) — meson terms ~2x too small vs rho_s — a
convention/normalization mismatch. Two things to nail before shipping any set:
1. **Verify each set's couplings against its own saturation point** (run
   `saturation_properties` and require it reproduces the published n0, B/A, K, m*/M,
   E_sym, L) before exposing it.
2. **c < 0 sets** (NL3 and many modern parameterizations) have a nonlinear scalar
   potential **unbounded below** → the physical EOS is a *metastable* branch; the
   first-root pick is correct *if* the root exists, but selection/robustness for
   these needs explicit care.

**Why deferred:** GM1 (default) is fully validated, so the engine is proven; this is
about breadth of parameter sets, best done with a reference table in hand.

## Enthalpy-parameterized, multi-branch EOS inference (RIFT framing)

**What:** RIFT's tabular EOS inference (and our drop-in legacy emitter) reduce each
EOS to a **single stable branch** ordered by Lambda(m_ref), then sample that 1-D
ordering and look up Lambda(m1), Lambda(m2). This **throws away disconnected stable
branches** (twin stars from a strong first-order phase transition) and forces a
single-valued M->Lambda map.

**Better framing (RO):** the low-level sequence parameter is the **central
pseudo-enthalpy h_c**, which is monotone. Parameterize the inference *directly* along
h_c (a path integral in enthalpy) rather than along Lambda(m_ref). M(h_c), R(h_c),
Lambda(h_c) are single-valued in h_c, so all branches — including disconnected twin
branches — are represented natively; the binary's component masses select the
branch(es) consistently, and the likelihood integrates over h_c (per star) instead of
over a collapsed 1-D Lambda ordering.

**Why deferred / status:** the new `NSSequence` format already stores everything vs
h_c (the substrate for this), and `write_landry_legacy` keeps today's RIFT working
(single branch). Implementing the enthalpy-path-integral likelihood is a change to
the RIFT inference driver (`util_ConstructIntrinsicPosterior_GenericCoordinates.py`)
+ the EOS-marginalization, which we can develop against our own source copy. Pairs
naturally with the twin-star/`build_family` branch bookkeeping item.

## lalsuite NS-sequence structure: drop-in / comparison

**What:** recent lalsuite moved toward an NS-sequence data structure (analogous to our
`NSSequence`). We should add a reader/writer pairing so the same generated sequence can
feed both RIFT (via `EOSSequenceNMB` / legacy `EOSSequenceLandry`, done) and the
lalsuite NS-sequence path, enabling cross-code comparison of M(h_c), R, Lambda on
identical EOS input.

**Status / hook:** the representation abstraction is in place
(`rift.SequenceRepresentation` + `open_sequence` dispatching on the HDF5
`representation` attr), so a `lalsuite_nsseq/1` representation can register as another
backend without touching consumers. Pin the exact lalsuite struct/fields before
implementing the adapter.

**Compression / functional representations (design mandate):** the same abstraction is
the seam for moving off pure tabular draws to *compressed / functional* internal
representations — empirical/spectral interpolants, PCA, or a Gaussian process over the
EOS function space (modelling functional spaces rather than draws). These register as
new `representation` tags (e.g. `gp_hc/1`, `pca_hc/1`) implementing the same
`SequenceRepresentation` accessor protocol; `write_nss`/`open_sequence` already carry
and dispatch on the tag. No consumer (RIFT, generator CLI) changes when the storage
changes.

## Muons in beta-equilibrium

**What:** beta-equilibrium NS matter contains muons once mu_e > m_mu (~106 MeV,
around n ~ 0.1-0.2 fm^-3). Currently only electrons are included (RMFT and
metamodel). Add muons via the common lepton chemical potential mu_e = mu_mu with
charge neutrality n_p = n_e + n_mu (an inner solve for mu_e given the charge
density). Small but real effect on the symmetry-energy-driven part of the EOS and on
Lambda/R at the ~0.1-1% level.

