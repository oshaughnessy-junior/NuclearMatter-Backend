# NuclearMatter-Backend — architecture & roadmap

*Design record, v0.0.1 (2026-06-02).  Author: R. O'Shaughnessy + collaborators.*

## 1. Purpose

Provide a single, modern, **differentiable** backend for neutron-star matter that
serves three inference contexts now handled by ad-hoc, lalsuite-bound code:

1. **"Normal" RIFT** — evidence for a single event *conditioned on a specific EOS*,
   and hence the per-EOS tabular **hyperpipeline** (`rift-eos-inference-tabular-method`).
2. **gwkokab** — population inference with low-dimensional EOS parameters.
3. **Nonparametric / GP** EOS inference (Landry–Essick / `RIFT-GP`) as a sibling
   representation.

The unifying requirement from `human_planning_notes.md`: a **robust, AD-capable TOV
solver** (multiple branches; parameterized by central enthalpy/entropy with
`dm/dh`, `dh/dm`; room for NN surrogates and rotating NS), fed by **modern EOS**
(tables *and* physical RMFT-style models), all credited cleanly (`ATTRIBUTION.md`).

## 2. Layered design

```
            ┌─────────────────────────── adapters ───────────────────────────┐
            │  RIFT EOSConcrete   │   gwkokab numpyro Dist   │  GP / tabular   │
            └───────────▲─────────────────────▲────────────────────▲──────────┘
                        │                      │                    │
                ┌───────┴──────────────────────────────────────────┴───────┐
   sequences →  │  tov.family : M-R-Λ, stable/unstable branches, dM/dh_c   │
                └───────────────────────────▲──────────────────────────────┘
                                             │  (backend seam: solve_structure)
                ┌────────────────────────────┴─────────────────────────────┐
   structure →  │  tov.solver : JAX RK4 TOV-in-enthalpy + tidal y → k2, Λ   │
                │   alt backends: RePrimAnd (ref) · Zhu C++ δ-TOV (future)  │
                └────────────────────────────▲─────────────────────────────┘
                                             │  EOSTable(n, eps, p, h, cs2)
                ┌────────────────────────────┴─────────────────────────────┐
   matter   →   │  eos.* : metamodel (RMFT-style) · tabular/CompOSE · poly  │
                └──────────────────────────────────────────────────────────┘
                                             │
                              constants/units (geometric km ↔ MeV/fm³ ↔ SI)
```

Two interfaces are the contract that keeps the layers decoupled:

- **`EOSTable`** (`eos/base.py`): JAX arrays `(n, eps, p, h, cs2)` in geometric
  units, with `at_h(h)` differentiable interpolation.  *Any* EOS — physical model,
  CompOSE table, GP draw — reduces to this, so the TOV layer never knows the source.
- **`solve_structure(table, h_c) → TOVPoint`** (`tov/solver.py`): the **backend
  seam**.  Callers depend only on this signature, so a high-accuracy reference
  (RePrimAnd) or a compiled forward-sensitivity engine (Zhu C++ δ-TOV) can replace
  the JAX integrator without touching adapters.

## 3. Key technical decisions

**Pseudo-enthalpy as the integration variable (Lindblom 1992).** Removes the
surface coordinate singularity, makes the central boundary a simple series start,
and gives a clean low-dimensional state `(r, m, y)`.  Central density is then
parameterized by `h_c`, directly satisfying the "model vs central variable, not
mass" requirement; `tov/family.py` exposes `dM/dh_c` (by AD) and `dh_c/dM`.

**JAX `lax.scan` RK4, forward-mode AD.** The whole `params → EOS → TOV → observable`
chain is `jit`/`vmap`/`grad`-able. We use **forward-mode** sensitivities: efficient
for the small EOS parameter count and *mathematically identical* to the hand-derived
perturbation ("δ-TOV") equations (cf. Zhu `delta_tt`). This is why we chose a
JAX-native solver over an FFI bridge to compiled C++: in an inference loop the
throughput bottleneck is `jit`/`vmap`/GPU batching of the *whole* likelihood
(gwkokab is JAX), and an FFI boundary per sample would break that fusion. The C++
δ-TOV remains valuable as a fast specialized backend behind the seam once Z. Zhu is
formally a repo developer and the code is licensed.

**Tidal integration is optional.** Reverse-mode `grad` seeds a zero cotangent on
`y`, whose near-center `1/r²` terms then give `0*inf → nan`. `solve_structure(...,
tidal=False)` drops `y` so reverse-mode `R, M` gradients are clean; forward-mode
handles `Λ`. Documented in README and tests.

**Physical EOS = nuclear empirical parameters + causal tail.** `eos/metamodel.py`
Taylor-expands the isoscalar/isovector energy around saturation (NEPs: `E_sat,
K_sat, Q_sat, E_sym, L_sym, K_sym, Q_sym`), solves β-equilibrium with electrons,
and gets pressure via the **envelope theorem** (energy is stationary in the proton
fraction, so `p = n dε/dn` is taken at fixed `x_p`). Because the bare Taylor series
softens unphysically above ~2 n_sat, we append a **constant-sound-speed causal
extension** (`cs2_tail`), as in `jester`'s MetaModel+CSE — giving a physical
stiffness/`M_max` knob for inference.

## 4. What is validated (v0.0.1)

`pytest` (17 tests): units; EOS monotonicity & causality; a physical star
(`M_max≈2.11 M⊙`, `R(1.4)≈11.7 km`, `Λ(1.4)≈449`); stable→unstable branch
turnover; `Λ` decreasing with mass; stiffness knob raises `M_max`; RIFT-adapter
surface; and **forward-mode AD vs finite difference** agreeing to `<1%` on `M_max`
and `~1e-4` on `R` (the smooth observables). `Λ` sensitivities are AD-exact but
numerically noisier pending a proper crust + adaptive stepping.

## 5. Roadmap

| Area | Next step |
|---|---|
| Crust | Replace the low-density metamodel continuation with a real SLy/BPS table; smooth match. Tightens `R`, `Λ`. |
| EOS | CompOSE/tabular ingestion (`EOSFromTabularData` parity); self-consistent **RMFT Lagrangian** solver (σ-ω-ρ) as a sibling `EOSModel`; reuse MUSES modules. |
| Solver | Optional **diffrax** adaptive backend; RePrimAnd reference cross-check; Zhu **C++ δ-TOV** backend behind the seam; **rotating NS** (slow-rotation → full RNS, ref 2604.05428). |
| Surrogates | NN surrogates for `M-R-Λ` and for `dm/dh, dh/dm` (Zhu 2604.05428 / 2604.24949). |
| RIFT | Wire the adapter into `util_RIFT_pseudo_pipe.py` tabular path; emit `EOSSequenceLandry`-format HDF5 for the hyperpipeline. |
| gwkokab | `numpyro` Distribution exposing EOS params + selection-function `V_T(EOS)`; joint population+EOS likelihood. |
| GP | Gaussian-process `φ(h)` representation feeding the same `EOSTable`/TOV path (RIFT-GP parity). |

## 6. Provenance

See `ATTRIBUTION.md`. Briefly: formulation motivated by `jester` (MIT) and the
nuclear metamodel literature (Margueron 2018); tidal `y` equations from Hinderer
2008 / Postnikov 2010; δ-TOV/RMF and Fisher-matrix application motivated by Z. Zhu's
`delta_tt` / `fisher_eos` (no license yet → motivated-by only, no code copied);
RIFT `EOSConcrete` is the target interface; gwkokab (Apache-2.0) is a downstream.
