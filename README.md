# NuclearMatter-Backend (`nmbackend`)

A differentiable (JAX) neutron-star **equation-of-state + TOV** toolkit for EOS
inference, designed to interface with **RIFT** (per-EOS evidence and the tabular
hyperpipeline) and **gwkokab** (population inference).  It updates and extends the
preliminary EOS tooling in RIFT's `EOSManager.py`.

> Status: **v0.0.1 — research scaffold + one validated vertical slice.**
> See `docs/architecture.md` for the full design and roadmap, and `ATTRIBUTION.md`
> for provenance/licensing of the works this is based on or motivated by.

## What works today

A single end-to-end, **fully differentiable** path:

```
nuclear params  ─►  physical EOS (metamodel + causal tail)  ─►  AD TOV solver
                                                                   │
                          M(h_c), R(h_c), Λ(h_c), branches, dM/dh_c
                                                                   │
                                            ┌──────────────────────┴───────────┐
                                       RIFT EOSConcrete adapter        forward-mode
                                       (lambda_from_m, P(ρ), mMax)     sensitivities
```

Validated numbers for the default parameters (`pytest`, 20 tests):
`M_max ≈ 2.08 M⊙`, `R(1.4) ≈ 11.7 km`, `Λ(1.4) ≈ 443`, `dR/dL_sym ≈ +0.027 km/MeV`
(reproducing the known L_sym–R(1.4) correlation).  The adaptive solver is
cross-validated against an independent radius-domain reference to **<0.1%** in
M, R, and Λ (see Validation status below).

## Install / run

```bash
pip install -e .            # core deps: numpy, scipy, jax
PYTHONPATH=src python examples/quickstart.py
PYTHONPATH=src pytest        # 17 tests
```

## Quickstart

```python
import jax
from nmbackend.eos import MetaModelEOS
from nmbackend.tov import build_family, m_max, radius_of_mass, lambda_of_mass
from nmbackend.tov.solver import solve_structure
from nmbackend.adapters import EOSConcreteNMB

mm = MetaModelEOS()
params = dict(K_sat=230., E_sym=32., L_sym=60., cs2_tail=0.4)

fam = build_family(mm.eos_table(params))         # M-R-Λ sequence (+ branches)
m_max(fam); radius_of_mass(fam, 1.4); lambda_of_mass(fam, 1.4)

# forward-mode AD sensitivity (the delta-TOV / Fisher building block)
hc = 0.20 * float(mm.eos_table(params).h[-1])
jax.jacfwd(lambda L: solve_structure(
    mm.eos_table({**params, "L_sym": L}), hc).R_km)(60.)

# drop into RIFT
eos = EOSConcreteNMB(mm, params)                 # .lambda_from_m, .pressure_density_on_grid, .mMaxMsun
```

## Layout

```
src/nmbackend/
  constants.py            units (geometric km; MeV/fm^3 <-> km^-2; SI for RIFT)
  eos/
    base.py               EOSModel ABC, EOSTable, pseudo-enthalpy build
    metamodel.py          nuclear empirical-parameter EOS + causal (CSS) tail
    polytrope.py          analytic polytrope (solver validation)
  tov/
    solver.py             TOV-in-enthalpy + tidal y -> k2, Λ; backends: diffrax
                          (adaptive, default) and scan (fixed-step) behind one seam
    family.py             M-R-Λ sequences, stable/unstable branches, dM/dh_c, dh_c/dM
  adapters/
    rift.py               RIFT EOSConcrete-compatible adapter
  reference/
    scipy_tov.py          independent radius-domain SciPy reference (cross-check)
    reprimand.py          RePrimAnd reference adapter (optional; pyreprimand)
tests/                    units, EOS, TOV, RIFT adapter, AD-vs-FD, reference-validation
examples/quickstart.py    scripts/validate_against_reprimand.py
env/                      pixi.toml + environment.yml for the validation env
docs/                     architecture.md, validation-plan.md, environment.md
```

## Gradients: use forward-mode

The solver is differentiable.  Use **forward-mode** AD (`jax.jacfwd`, `jax.jvp`)
for sensitivities of any TOV output — it is efficient for our small parameter
count and reproduces the perturbation ("delta-TOV") equations.  Reverse-mode
(`jax.grad`) is supported for **structure-only** solves (`tidal=False`, giving
`R`, `M`); reverse-mode through the tidal variable is not recommended (near-center
`1/r²` terms give `0*inf` in the backward pass).

## Validation status (important)

Cross-validation already changed the design. An independent radius-domain reference
(`nmbackend.reference.scipy_tov`, SciPy RK45 + (H,β) tidal) showed the original
fixed-step RK4-in-`h` default was under-resolved (**M off ~5%, Λ off ~25%**), so the
default is now an **adaptive diffrax (Dopri5+PID) backend**, which agrees with the
reference to **<0.1%** in M, R, Λ and carries clean forward-mode AD
(`tests/test_validation_reference.py`).

Still **open** (do not trust for these yet): **strong first-order phase transitions /
soft segments where cs² → 0**. There `dh=dp/(ε+p)` collapses, the tidal term
`(ε+p)/cs²` blows up, and the EOS table currently clips cs² and `cummax`-smears
pressure — wrong for a real hybrid EOS. These need the tidal junction condition and
an external **RePrimAnd**/**Zhu δ-TOV** cross-check (the sandbox can't build them; see
[`docs/environment.md`](docs/environment.md) to run them on your workstation, and
[`docs/validation-plan.md`](docs/validation-plan.md) for the ladder + metrics).

## Roadmap (see `docs/architecture.md`)

CompOSE/tabular ingestion · proper SLy/BPS crust · self-consistent RMFT Lagrangian
solver · NN surrogates (Zhu 2604.05428) · rotating NS (RNS) · gwkokab population
adapter · GP/nonparametric EOS · optional RePrimAnd and Zhu C++ δ-TOV backends.
