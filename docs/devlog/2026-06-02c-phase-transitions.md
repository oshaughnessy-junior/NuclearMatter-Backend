# Dev log — 2026-06-02 (c) — first-order phase transitions + tidal junction

**Who:** R. O'Shaughnessy (with AI pair).  Follow-on to `...b-validation-adaptive-solver`.

## Goal
Address the open item: strong first-order phase transitions (cs²→0 / density
discontinuities), which the smooth solver mishandles and which need the tidal
junction condition.

## Built
- `eos/hybrid.py` — `HybridEOS`: hadronic metamodel up to `p_trans`, a Maxwell
  density jump (μ-continuity sets the quark-side baryon density), then a CSS quark
  branch.  Exposes smooth segment sub-tables + `TransitionInfo`, a combined table,
  and a finite-width "smeared" table for limit checks.
- `tov/phase_transition.py` — `solve_hybrid`: integrates each phase as a separate
  adaptive (diffrax) segment and applies the tidal junction at the discontinuity,
      y_out = y_in − 4π r³ (ε_out − ε_in)/(m + 4π r³ p_t)
  (Damour–Nagar 2009; Postnikov 2010; Takátsy–Kovács 2020).
- `reference/scipy_tov.py` — `solve_reference_hybrid`: an *independent*
  radius-domain (H,β) solver with the junction (Δβ = (H/r)Δy), as the cross-check.

## Validation
- `solve_hybrid` vs `solve_reference_hybrid` (two independent formulations, both with
  the junction): **<0.1% in M, R, Λ** across the hybrid sequence incl. quark-core
  stars.  This corroborates the junction *code/algebra/sign*.
- Hybrid sequence is physical: hadronic stars up to ~1.59 M⊙ at the transition, then
  quark-core stars with smaller R and Λ (transition softening).
- Smeared-crossover → sharp limit: the junction value sits in the right ballpark
  (~0.4% at a well-resolved width at fixed h_c).  A *clean* monotone smeared test is
  hampered by under-resolution at tiny width and by twin-star non-monotonicity of
  M(h_c) at fixed mass — hence the (H,β) reference is the decisive in-sandbox check.
- 24 tests pass (added `tests/test_phase_transition.py`, 4 tests).

## Honest limits
- The junction *formula* is shared by both solvers, so this validates the
  implementation, not the formula derivation.  RePrimAnd remains the external arbiter
  (`scripts/validate_against_reprimand.py`, run on the workstation).
- `build_family`/the RIFT adapter do **not** yet split twin-star/disconnected
  branches; `solve_hybrid` is per-`h_c`.  Hybrid construction is NumPy (no AD through
  transition params yet).

## Next
Twin-branch-aware family + adapter for hybrid EOS; SLy/BPS crust; run the RePrimAnd
and Zhu δ-TOV cross-checks in the workstation env.
