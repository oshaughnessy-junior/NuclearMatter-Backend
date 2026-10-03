# Core transport extension on merged C2

2026-10-03. The CIT C2 implementation is merged through PRs #21–25, main `f97b5fd`. Work here builds on those gap, emissivity, heat-capacity, conductivity and cooling-curve modules. The earlier standalone heat-capacity and cooling-quadrature prototypes should not become parallel production implementations.

## Implemented

`nmbackend.cooling.physics.core_conduct.lambda_core` implements the NSCool 2009 / Shternin–Yakovlev 2007 electron–muon conductivity, including proton pairing and the source's corrected scattering suppression fit. It takes local temperature, electron/muon/proton Fermi momenta, proton Landau mass ratio and proton critical temperature. It broadcasts under JAX, preserving the source's absent-species branches and single-precision exponent constants.

The assembled star now returns core and crust conductivity. The last inward crust-blanket step uses core conductivity and the actual proton gap at `icore`, replacing the previous crust proxy. The internal crust-march interface requires `tc`; deeper-core marching remains outside this blanket approximation.

The scalar reference comparison caught a small error at exactly T=Tc caused by the shared gap helper's clipping. The port now preserves the reference value there while keeping inactive square-root branches safe. Transition derivatives remain excluded.

## Validation on cardassia

22 focused tests pass: six new core-conductivity checks and all 16 existing assembled-star checks. A separate run without NSCool confirms that the standalone AD test still runs (one passed, five reference cases skipped). No dependency was added.

| APR mass and gaps | Worst relative core-conductivity error | Worst relative inner-boundary temperature error |
|---|---:|---:|
| 1.4, none | 9.97e-9 | 1.75e-8 |
| 1.4, default | 9.53e-8 | 1.64e-8 |
| 2.0, Zhou | 3.68e-8 | 1.60e-8 |
| 2.0, CCDK with extended proton gap | 8.70e-9 | 1.54e-8 |

Reference values come from NSCool's compiled scalar routine and its zone dumps. The paired-boundary result and gradient agree between 20 and 40 Picard iterations. Its gradient with respect to log temperature, envelope parameter and proton-gap amplitude also agrees with central finite differences, including the reverse path through inactive conductivity branches.

A composed late-time cooling diagnostic uses a 241-point temperature grid and a fixed 1 kyr thermal state from the baseline calculation. For APR 1.4, it propagates envelope, proton-gap and proton-mass changes through the actual JAX rates, heat capacity, conductivity and envelope together. At 10 kyr:

| Parameter | d ln Teff / d ln parameter |
|---|---:|
| PCY97 envelope eta | 0.02042 |
| Proton gap amplitude | -0.002662 |
| Proton Landau mass scale | 0.02310 |

AD agrees with central finite differences within 2.44e-11 absolute. This validates derivatives of the discretized, fixed-background model, not observational accuracy or parameter identifiability. The baseline and perturbed trajectories are explicitly gated for interpolation coverage. The anchor is supplied from the model, so this is not an independent transport validation.

## Scientific scope

NSCool deliberately returns only the lepton conductivity for ICON_CORE=2, even though it calls a baryon routine. Neutron conduction can matter; this is a named reference prescription, not a universal total-core model. [Shternin & Yakovlev 2007](https://arxiv.org/abs/0705.1963).

The superconducting screening uses an older Pippard-limit fit. The improved treatment should enter as a separately validated alternative, especially for larger gaps and lower densities. [Shternin 2018](https://arxiv.org/abs/1805.06000).

Stellar structure and the thermal handoff remain fixed. Full nuclear-parameter sensitivities still need the backend TOV/composition-to-Star connection and early-time transport. Magnetic anisotropy, hyperons and a rotating thermal-evolution solver are not implemented by this change.

## Reproduction and coexistence

Worktree: `/home/oshaughn/rotating-ns-feasibility/core-transport`; branch: `codex/cooling-core-conductivity`, based on merged main `f97b5fd`. Previous prototypes and CIT checkouts were left intact. Upstream was checked again after the PR merges; there was no overlapping core-transport implementation.

Use the existing NSCool builder in this branch. Our separate installation is `/home/oshaughn/rotating-ns-feasibility/nscool-c2/NSCool`; no reference source is vendored. With `NSCOOL_HOME` set to its parent prefix and `PYTHONPATH=src`:

    python -m pytest tests/test_cooling_core_conduct.py tests/test_cooling_physics_star.py -q
    python examples/core_conduct_validation.py /path/to/results

Raw measurements: local `core-conduct-results.json`; the PR stores the same measurements in `docs/cooling/core_transport_results.json`. Science, numerical and code-style adversarial reviews checked the source prescription, model limitations, species and pairing branches, boundary indexing, inactive-branch AD, iteration convergence and test scope. Their requested corrections were incorporated.
