# Dev log — 2026-06-02 — initial scaffold + first vertical slice

**Who:** R. O'Shaughnessy (with AI pair).  **Repo:** `NuclearMatter-Backend`.

## Goal for the session
Stand up `nmbackend` and one **validated, fully differentiable** end-to-end path:
`nuclear params → physical EOS → AD TOV → M/R/Λ → RIFT EOSConcrete adapter`.
Scope agreed: (1b) scaffold + vertical slice; TOV engine = JAX/diffrentiable
(forward-mode = δ-TOV sensitivities), with a documented seam for Zhu's C++ δ-TOV
later; (3a) RIFT adapter first; (4a) physical nuclear model first.

## What was built
- Package `src/nmbackend`: `constants` (geometric-km units), `eos/` (base +
  `MetaModelEOS` + `PolytropeEOS`), `tov/` (`solver`, `family`), `adapters/rift`.
- JAX `lax.scan` RK4 **TOV-in-pseudo-enthalpy** with in-loop tidal `y` → `k2`, `Λ`.
- Nuclear **metamodel** (NEP Taylor expansion + β-equilibrium electrons + envelope-
  theorem pressure) with a **constant-sound-speed causal tail** (`cs2_tail`) so it
  stays stiff/causal above ~2 n_sat.
- M–R–Λ **family** with stable/unstable branch detection and `dM/dh_c`, `dh_c/dM`.
- **RIFT EOSConcrete adapter** (`lambda_from_m`, `lambda_from_m_vector`,
  `pressure_density_on_grid`, `mMaxMsun`).
- 17 `pytest` tests incl. **AD-vs-finite-difference** validation.

## Results (default params)
`M_max ≈ 2.11 M⊙`, `R(1.4) ≈ 11.7 km`, `Λ(1.4) ≈ 449`, `dR/dL_sym ≈ +0.027 km/MeV`
(correct sign/scale of the L_sym–R correlation). All 17 tests pass.

## Key decisions / gotchas
- **Forward-mode AD** is the supported path: it equals the perturbation/δ-TOV
  equations and avoids the reverse-mode `0*inf` from the tidal variable's near-
  center `1/r²` terms. `tidal=False` gives clean reverse-mode `R, M` for inference.
- β-equilibrium proton fraction solved by bisection + **one Newton step** so AD
  recovers the correct implicit `dx_p/dparam`.
- Bare Taylor metamodel softens unphysically at high n → added the CSS tail
  (stiffness becomes an inference knob).
- Λ sensitivities are AD-exact but numerically noisy without a real crust; tests
  validate AD tightly on the smooth observables (`M_max`, `R`) and check Λ to the
  tidal noise floor.

## Attribution notes (see ATTRIBUTION.md)
- **Motivated by** `jester` (MIT, P. Pang) for the enthalpy formulation & metamodel,
  and the Margueron 2018 metamodel; tidal `y` from Hinderer 2008 / Postnikov 2010.
- **Motivated by only** (no code copied): Z. Zhu `delta_tt` + `fisher_eos` (no
  LICENSE present). Plan: add a compiled C++ δ-TOV backend behind the solver seam
  once Z. Zhu is a formal repo developer and a license is set.
- RIFT `EOSConcrete` is the target interface; gwkokab (Apache-2.0) is downstream.
- New, MIT-licensed: the JAX scan-RK4 enthalpy solver, the envelope-theorem
  metamodel, the branch-aware family, the RIFT adapter.

## Next
Real SLy/BPS crust → CompOSE ingestion → RMFT Lagrangian solver → wire adapter into
`util_RIFT_pseudo_pipe.py` tabular path → gwkokab numpyro Distribution. (docs/architecture.md §5)
