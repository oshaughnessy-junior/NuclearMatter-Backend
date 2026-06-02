# NuclearMatter-Backend — forward roadmap

*v0.0.1 (2026-06-02).  Sequencing per RO: harden the solver first, then add physics
(RMFT), then wire to RIFT.  Proof-of-concept is the milestone; surrogates only if we
need the acceleration.*

Status today: a differentiable JAX TOV solver (adaptive diffrax default) + a
nuclear metamodel EOS + a Maxwell phase-transition path with the tidal junction,
all cross-validated **internally** to <0.1% (diffrax vs an independent radius-domain
(H,β) reference) and now driving **RePrimAnd 1.7** as an external reference.

The external check surfaced the headline science issue:

> **RePrimAnd vs ours disagree by ~200 m in R in some cases** (~1–2% at R≈11–12 km),
> far above our internal diffrax-vs-SciPy agreement (~2 m, 2e-4).  This is the
> accuracy gap to close before the backend is trustworthy for precision EOS
> inference.

## 0. Accuracy target (why hardening comes first)

3G detectors (ET/CE) and **stacked / population** measurements drive the radius and
Λ *statistical* precision toward the sub-percent / tens-of-meters level.  Solver and
EOS-representation **systematics must sit an order of magnitude below** that — call
it a budget of **≲10 m in R** and **≲0.5% in Λ** for the numerics alone.  The current
~200 m RePrimAnd-vs-ours discrepancy is ~20× over budget, so this is the gate.
We already have the adversarial machinery (independent (H,β) reference, the
phase-transition junction tests, RePrimAnd, and — newly — MUSES `qlimr`) to drive
it down quantitatively, which is exactly why hardening is cheap to do now and
expensive to defer.

## Phase 1 — Harden the solver (FIRST)

Goal: bring the multi-solver M–R–Λ(–I) spread under the §0 budget across a ladder of
EOS, and understand *every* residual.

1. **Diagnose the ~200 m R discrepancy.**  Likely contributors, in rough order:
   - *EOS representation.*  RePrimAnd rebuilds its own monotone spline from our table
     and integrates in its own enthalpy; we currently use **log-linear `jnp.interp`**
     and **clip cs² / `cummax`-smear pressure** when building the table.  R is most
     sensitive at low density, where these differ most.  → switch to a
     thermodynamically consistent, monotone (PCHIP/cubic) reconstruction; stop
     clipping cs²; derive cs² analytically where the EOS allows.
   - *Surface definition.*  We stop integrating at the table's `h_min` (lowest
     tabulated density); RePrimAnd integrates to its own surface.  A mismatched
     surface is a ~100 m R effect.  → define a single, shared surface convention and
     a consistent low-density continuation.
   - *Baryon-mass / unit conventions* (we already match RePrimAnd's 1.66e-27 kg;
     verify end-to-end with a units round-trip test).
2. **Reference matrix + error tables.**  One harness that runs, on the same EOS:
   our diffrax solver, the SciPy (H,β) reference, RePrimAnd, MUSES `qlimr`, and
   lalsim (named EOS).  Emit per-quantity (M, R, Λ, M_b, I) relative-difference
   tables vs the most accurate reference.  Track against the §0 budget.
3. **Convergence study.**  Richardson in diffrax tolerances *and* in EOS-table
   resolution; confirm RK/Dopri order on smooth EOS and bounded error across
   transitions.  Publish the curves.
4. **Adversarial EOS ladder** (extend `tests/`):
   analytic (Tolman VII / constant-density, closed-form k₂) → smooth tabulated
   (SLy/APR/MPA1 via lalsim) → metamodel → **strong first-order transition + twin
   stars** (already have the junction; add the RePrimAnd cross-check) → multi-segment.
5. **Tidal sector.**  Validate Λ and k₂ to the §0 budget, including the discontinuity
   junction, against RePrimAnd and `qlimr`.
6. **Moment of inertia / slow rotation** (cheap, high-value): add I(M) and the
   I–Love–Q quasi-universal check (`qlimr` is the natural reference); a strong
   internal consistency test that catches metric-sector bugs.

Exit criterion: multi-solver spread ≤ budget on the smooth + metamodel ladder, and
characterized (not necessarily ≤ budget) on the transition ladder, with every
residual attributed.

## Phase 2 — Add physics: RMFT (after hardening)

Keep the fast **NEP metamodel** as the lightweight inference model; add a faithful
**relativistic mean-field** sibling under the same `EOSModel`/`EOSTable` interface.

1. **Self-consistent RMFT Lagrangian solver** (σ-ω-ρ with nonlinear self-couplings;
   optional δ, hyperons).  Solve the meson field equations + β-equilibrium at each
   density (differentiable root-find, as we already do for the proton fraction).
2. **Reuse MUSES `cmf_solver`** (Chiral Mean Field — RMFT-class, with hyperons) as a
   *reference EOS generator* and a validation target; cross-check our RMFT against it.
   MUSES `eos_taylor_4d` validates the metamodel branch; `njl`/quark modules feed the
   hybrid path; `qlimr` is a TOV/observables reference for Phase 1.
   **Integration model:** MUSES is a containerized Django/Celery stack (PostgreSQL +
   RabbitMQ + Celery + MinIO), brought up via Docker (`docker/launch.sh`, API on
   :4000) — see `2026-04-22-Explore-MUSES/`.  We interface with it *out of band*:
   trigger module workflows via the API and **mount the resulting CompOSE-format
   tables** (from the MinIO/volume outputs) into the backend.  It is an EOS
   *provider* and reference, never in the inference hot path.
3. **CompOSE / tabular ingestion** to parity with `EOSFromTabularData`, so any MUSES-
   or community-generated table flows through the hardened solver.
4. Re-run the Phase-1 reference matrix on the RMFT and tabular EOS.

## Phase 3 — Wire to RIFT (after physics)

1. Emit `EOSSequenceLandry`-format HDF5 (per-EOS precomputed M–R–Λ) so the new
   backend drops into the existing tabular hyperpipeline.
2. Thread the `EOSConcreteNMB` adapter through `util_RIFT_pseudo_pipe.py`'s
   `--internal-tabular-eos-file` path; reproduce a GW170817 per-EOS evidence run.
3. gwkokab adapter (numpyro Distribution over low-dim EOS params + selection
   function); GP/nonparametric representation feeding the same `EOSTable`/TOV path.

## The crust (explicit caveat)

A correct, *consistent* crust (matching the core EOS at the crust–core transition,
with the right composition and the inner-crust pasta) is genuinely hard and is a
real systematic for R and Λ.  Plan: (a) treat the crust as a **documented systematic
budget**, not a solved problem; (b) use a unified crust–core table where the model
provides one (MUSES `crust_dft`), else a standard SLy/BPS outer+inner crust matched
at a transition density; (c) quantify the crust-induced R/Λ spread as part of the
Phase-1 error budget and, for inference, consider marginalizing over crust choice
rather than fixing it.

## Proof-of-concept milestone

After Phases 1–3 we have a hardened, physically-grounded, RIFT-wired backend = the
intended proof of concept.  **Surrogates are deferred** and only pursued if the
hardened solver is too slow in the inference loop: NN surrogates for M–R–Λ(–I) and
for `dm/dh`, `dh/dm` (Zhu arXiv:2604.05428 / 2604.24949), trained on the hardened
solver and validated against it on held-out EOS.  Rotating NS (RNS) sits with the
surrogate/acceleration work.

## Near-term next actions

1. Build the Phase-1 reference matrix harness (diffrax / SciPy(H,β) / RePrimAnd /
   `qlimr` / lalsim) and the error tables.
2. Replace log-linear interpolation + cs² clipping with a monotone,
   thermodynamically consistent EOS reconstruction; unify the surface convention.
3. Re-measure the RePrimAnd-vs-ours R/Λ spread; attribute and drive toward ≲10 m.
