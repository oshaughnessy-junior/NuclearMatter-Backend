# TOV validation plan — accuracy, phase transitions, and adaptive stepping

*v0.0.1 (2026-06-02).  Owner: R. O'Shaughnessy + collaborators.*

> The v0.0.1 solver passes *sanity* tests (causal, branch turnover, sensible
> M–R–Λ, AD≈FD on smooth observables).  Sanity is **not** accuracy.  This document
> defines how we earn confidence by cross-validating against independent reference
> solvers, with explicit attention to the failure mode the fixed-step integrator is
> most exposed to: **phase transitions / soft segments where cs² → small or 0.**

## 1. Why this matters (the known weak spot)

We integrate the TOV + tidal system with a **fixed number of RK4 steps linearly
spaced in pseudo-enthalpy `h`** (`tov/solver.py`).  Two structural risks:

1. **Soft segments / first-order phase transitions.** Where `cs² = dp/dε → 0`
   (or a literal density discontinuity), `dh = dp/(ε+p)` collapses: a wide range of
   `ε` maps to a thin sliver of `h`.  Linear-in-`h` nodes then *under-sample* the
   transition, and the tidal equation has `(ε+p)/cs²` which **blows up** as `cs²→0`.
   The current table even *clips* `cs²` to `[1e-8, 1]` and applies `cummax` to `p`,
   which silently smears transitions — fine for a smooth metamodel, **wrong** for a
   real phase-transition EOS.  The uniform-matter models now reject a core whose
   raw pressure falls with density below the maximum-mass star
   (`eos.crust.monotone_core`); `allow_unstable_core=True` keeps the plateau table.
2. **Surface / crust.** Λ depends sensitively on the low-density tail; without a
   real crust and with a coarse surface, Λ carries ~1% numerical noise (already
   observed: FD on Λ jitters while AD returns the exact slope of the noisy function).

Conclusion: before trusting Λ (and M_max for strongly first-order EOS), we must
(a) cross-validate against reference solvers, and (b) very likely move to **adaptive
stepping** and **transition-aware handling**.

## 2. Reference solvers

| Reference | Use | Notes |
|---|---|---|
| **RePrimAnd** (`pyreprimand`; Zenodo 10617510) | Primary high-accuracy reference for M, R, Λ, M_b. | Already wrapped in RIFT `EOSManager.EOSReprimand` / `make_mr_lambda_reprimand`; gives a turnkey, well-tested TOV sequence with tidal. Reuse that path to generate truth. |
| **Zhu `delta_tt`** (C++ RK4 + perturbation δ-TOV) | Cross-check of M, R, Λ **and** of `dΛ/dparam` sensitivities. | Validates our forward-mode AD against an independent analytic-perturbation computation. (Motivated-by until licensed — coordinate with Z. Zhu.) |
| **lalsimulation** `SimNeutronStar*` | Secondary cross-check via RIFT `make_mr_lambda_lal`. | Convenient for built-in named EOS (SLy, AP4, MPA1…). |
| **Analytic** (Tolman VII, constant-density/Schwarzschild interior) | Closed-form unit tests independent of any code. | Exercises the integrator core with known M(R), and known k₂ limits. |

## 3. Test EOS ladder (increasing difficulty)

1. **Analytic**: incompressible/Tolman-VII — closed-form M(R), compactness limit.
2. **Smooth tabulated**: SLy4, APR, MPA1 (CompOSE / lalsim names).  No transitions.
3. **Our metamodel**: several NEP points spanning soft/stiff `cs2_tail`.
4. **Strong first-order phase transition**: a hybrid hadron–quark EOS with a
   Maxwell construction (explicit ε jump at constant p, `cs²=0` plateau) — e.g. a
   constant-speed-of-sound quark core grafted onto SLy.  **This is the stress test.**
   Include cases with a *disconnected* (third-family/twin-star) branch.
5. **Multi-segment / multiple transitions** to probe branch bookkeeping.

## 4. Acceptance metrics

For each EOS and a grid of central densities, compare our solver to the reference:

- **Bulk**: `|ΔM|/M`, `|ΔR|/R`, `|ΔΛ|/Λ` along the sequence.
  - Target (smooth EOS): `< 0.1%` in M,R; `< 1%` in Λ.
  - Target (phase-transition EOS): `< 0.5%` in M,R; `< 2–3%` in Λ *after* adaptive
    stepping (document pre-fix error to quantify the gain).
- **M_max and ρ_c(M_max)**: `< 0.2%` / `< 1%`.
- **Branch topology**: stable/unstable (and any twin-star) segments must match
  reference (sign of dM/dρ_c at the same central densities).
- **Tidal AD**: `dΛ/dL_sym`, `dΛ/dM` from forward-mode vs Zhu δ-TOV and vs
  Richardson-extrapolated finite differences: `< 1%` on smooth EOS.
- **Convergence/self-consistency**: Richardson study in `n_steps` (e.g. 400 → 6400);
  the solver must show the expected RK4 order on smooth EOS and *bounded* error
  across a transition (no order loss → diagnostic that stepping is inadequate).

## 5. Solver work this drives (likely outcomes)

- **Adaptive stepping**: replace fixed linear-`h` nodes with either
  (a) an adaptive embedded RK (Dormand–Prince) — natural via **diffrax**, which is
  JAX-AD-compatible (`diffrax.Dopri5` + PID controller; this is the `jester` path), or
  (b) an `h`-grid pre-refined around transitions (detect `cs²` minima in the table
  and add nodes), keeping the lightweight scan.
- **Transition-aware EOS table**: stop clipping/`cummax`-smearing; represent genuine
  discontinuities (Maxwell construction) explicitly, integrate the tidal **junction
  condition** at a density discontinuity (the Λ surface/interface term, Damour–Nagar
  / Hinderer: `k₂` gets a `-(...)·Δε/...` correction at each jump).
- **`cs² → 0` guard**: the `(ε+p)/cs²` term needs a principled treatment, not a
  clip; near a plateau the tidal equation is stiff and wants the adaptive controller.
- **Surface/crust**: attach SLy/BPS so Λ stops carrying crust noise.

## 6. Deliverables & sequencing

1. `tests/reference/` harness: `make_reference_sequence(eos, solver=...)` that runs
   RePrimAnd (and lalsim) via the RIFT helpers, caches truth tables (HDF5).
2. `tests/test_validation_smooth.py`: ladder steps 1–3 vs reference, metrics in §4.
3. Build the **phase-transition EOS** generator (`eos/hybrid.py`, Maxwell) and
   `tests/test_validation_transition.py` (ladder step 4–5).  Expect failures →
4. Implement adaptive stepping (diffrax backend behind the existing `solve_structure`
   seam) + transition handling; re-run until §4 targets are met.
5. Add the Zhu δ-TOV cross-check for AD sensitivities (once licensed).
6. Record results and the pre/post-adaptive error reduction in a dev-log entry.

## 7. Results so far (2026-06-02)

The plan already paid off.  Adding an **independent in-process reference**
(`nmbackend.reference.scipy_tov`: radius-domain SciPy RK45 + (H,β) tidal) immediately
revealed that the original **fixed-step RK4-in-h** default (`n_steps=800`) was badly
under-resolved: **M off ~5%, Λ off ~25%** — convergence was only ~1st order, the
error localized at the crude central boundary start, *not* the RK4 interior.

Fix: switched the default to an **adaptive diffrax (Dopri5 + PID) backend**, which
auto-resolves both stiff ends.  It now agrees with the SciPy reference to
**M < 0.05%, R < 0.02%, Λ < 0.1%** in ~250 adaptive steps, and forward-mode AD flows
through it (`diffrax.ForwardMode`), matching FD to ~1e-4.  The fixed-step scan is
retained as a lightweight backend and is shown to *converge* to the reference.
Tests: `tests/test_validation_reference.py`.

## 8. Status checklist

- [x] In-process independent reference (SciPy radius-domain + (H,β) tidal).
- [x] Adaptive (diffrax) backend behind `solve_structure`, validated to <0.1%.
- [x] Convergence study in `n_steps` (scan → reference).
- [x] RePrimAnd reference **scaffold** (`reference/reprimand.py`) + env files
      (`env/`, `docs/environment.md`) — runs on the user's workstation.
- [x] RePrimAnd reference adapter updated to the **modern (1.7) API**
      (`reference/reprimand.py`; same fix applied to RIFT `EOSManager`), with legacy
      fallbacks.  Cross-check ready to run in the validation env (`pip install
      reprimand` on Linux); execution on the user's workstation still pending.
- [ ] Analytic Tolman-VII / constant-density unit test.
- [ ] Smooth-EOS validation against lalsim named EOS (SLy/APR/MPA1).
- [x] Hybrid phase-transition EOS generator (Maxwell), `eos/hybrid.py` — hadronic
      + density jump + CSS quark branch; produces hybrid-star sequences.
- [x] Tidal **junction condition** at density discontinuities
      (`tov/phase_transition.solve_hybrid`), cross-validated against an independent
      radius-domain (H,β) junction solver (`reference.solve_reference_hybrid`) to
      **<0.1% in M, R, Λ**.  Junction formula: Damour–Nagar 2009 / Takátsy–Kovács 2020.
- [ ] Twin-star / disconnected-branch bookkeeping in `build_family` (the hybrid
      M(h_c) is non-monotonic; `solve_hybrid` is per-`h_c`, but the family/adapter
      do not yet split twin branches).
- [ ] Phase-transition cross-check against **RePrimAnd** (external arbiter).
- [x] SLy/BPS crust (`eos/crust.py`, default for the four uniform-matter models;
      R_1.4 pinned in `tests/test_crust.py`).
- [ ] Zhu δ-TOV cross-check of M/R/Λ and `dΛ/dparam`.

> Note: the sharp-jump+junction result is the width→0 limit of a finite-width
> crossover, but a clean monotone smeared-convergence test is hampered by
> under-resolution at tiny width and by twin-star non-monotonicity at fixed mass;
> the independent (H,β) junction reference is the decisive in-sandbox check.
