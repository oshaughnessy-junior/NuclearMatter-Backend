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

Validated numbers for the default parameters (`pytest`):
`M_max ≈ 2.08 M⊙`, `R(1.4) ≈ 11.7 km`, `Λ(1.4) ≈ 443`, `dR/dL_sym ≈ +0.027 km/MeV`
(reproducing the known L_sym–R(1.4) correlation).  The adaptive solver is
cross-validated against an independent radius-domain reference to **<0.1%** in
M, R, and Λ (see Validation status below).

## Install / run

```bash
pip install -e .            # core deps: numpy, scipy, jax, diffrax, pyyaml
PYTHONPATH=src python examples/quickstart.py
PYTHONPATH=src pytest        # full test suite
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
    base.py               EOSModel ABC, EOSTable, thermo-consistent at_h, surface
    metamodel.py          nuclear empirical-parameter EOS + causal (CSS) tail
    rmft.py               self-consistent sigma-omega-rho RMFT (GM1-validated)
    hybrid.py             Maxwell hadron-quark EOS (first-order phase transition)
    polytrope.py          analytic polytrope (solver validation)
  tov/
    solver.py             TOV-in-enthalpy + tidal y -> k2, Λ; backends: diffrax
                          (adaptive, default) and scan (fixed-step) behind one seam
    family.py             M-R-Λ sequences, stable/unstable branches, dM/dh_c, dh_c/dM
  lab_likelihood.py       density-tagged P_sym(n) lab likelihoods (MB-N2):
                          marginalizable transport-model systematic floor,
                          correlated (E_sym, L_sym) Gaussian, FRIB scenario ladder
  data/                   packaged copy (with provenance) of the FRIB scenario ladder
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

Accuracy of the adaptive (diffrax) tangents, AD vs finite differences
([devlog](docs/devlog/2026-10-12-tov-jump-ts.md)):

- **EOS-parameter derivatives** converge with `rtol` at the level of the values:
  ~3e-5 at the default `rtol=1e-7`, ~1e-7 at `1e-9`. Steps stop at every EOS
  table node (`jump_ts`), where the right-hand side has slope kinks. Before
  2026-10-12 they did not, and these tangents were off by up to 10%.
- **Central-enthalpy derivatives** (`d/dh_c`) converge like `rtol` but sit
  ~1e3x higher: ~1e-4 at the default, ~5e-7 at `1e-9`. Freezing the start of the
  integration is what makes them converge at all. The residual is a near-common
  factor on M, R and Λ, so it largely cancels in fixed-mass derivatives.
  `solve_structure(..., centre_refine=True)` cuts it ~10x for +40–90% cost.
- **First derivatives only in `h_c`.** `d²/dh_c²` (e.g. `jacfwd` of `dM/dh_c`)
  is wrong by orders of magnitude. Mixed `d²/dh_c dθ` is fine.
  Second derivatives in EOS parameters (`d²/dθ²`) are not supported either.

## Tidal term and node enthalpies (2026-10-14)

The tidal term `(ε+p)/c_s²` is the analytic `dε/dh` of the same Hermite cell
that gives `ε(h)`, and the table's node `h` comes from the log-linear `ε(p)` of
each cell (Gauss–Legendre) instead of the trapezoid
([devlog](docs/devlog/2026-10-14-tidal-cs2.md)). Before, the node `c_s²` left a
"phantom" energy jump at steep crust–core joins (Λ1.4 2–10% low; 6.2% on Q2C's
FSUGold2 join), and the trapezoid `h` put R1.4 4–10 m high on smooth tables.
Now Λ1.4 agrees with an independent radius-domain reference on the same nodes
to ≤ 6e-6 and R1.4 to < 0.1 mm on 18 test tables. Every Λ moves. `EOSTable.cs2` is now a
diagnostic only. `jump_nodes=False` is not recommended for tidal solves.

## Validation status (important)

Cross-validation already changed the design. An independent radius-domain reference
(`nmbackend.reference.scipy_tov`, SciPy RK45 + (H,β) tidal) showed the original
fixed-step RK4-in-`h` default was under-resolved (**M off ~5%, Λ off ~25%**), so the
default is now an **adaptive diffrax (Dopri5+PID) backend**, which agrees with the
reference to **<0.1%** in M, R, Λ and carries clean forward-mode AD
(`tests/test_validation_reference.py`).

**First-order phase transitions** are now handled: `eos/hybrid.py` builds a
Maxwell hadron–quark EOS (density jump at constant pressure), and
`tov.solve_hybrid` integrates each phase adaptively and applies the **tidal junction
condition** at the discontinuity (Damour–Nagar 2009 / Takátsy–Kovács 2020).  It
agrees with a radius-domain (H,β) solver that applies the same junction formula
to **<0.1%** in M, R, Λ; that checks the integration, not the formula.  Until
2026-10-14 both applied the junction with the wrong sign, and hybrid Λ was too
low: at fixed mass on `Hybrid(80, 250, 1)` by 0 / 0 / 0.5% / 9.9% / 18.6% at
1.2 / 1.4 / 1.6 / 1.8 / 2.0 M☉ (no quark core below ~1.57 M☉; M_max 2.06).  The sign is now
checked against the limit of the smeared transition, which needs no junction
(`tests/test_phase_transition.py`;
[devlog](docs/devlog/2026-10-14-tidal-cs2.md)). The hybrid sequence shows
the expected softening (quark cores → smaller R, Λ).

Still **open**: twin-star/disconnected-branch bookkeeping in `build_family` (the
hybrid M(h_c) is non-monotonic; `solve_hybrid` is per-`h_c` but the family/adapter
don't yet split twin branches); and the external
**RePrimAnd**/**Zhu δ-TOV** cross-checks (the sandbox can't build them — see
[`docs/environment.md`](docs/environment.md)). Plan + metrics:
[`docs/validation-plan.md`](docs/validation-plan.md).

## Roadmap (see `docs/roadmap.md` and `docs/architecture.md`)

Forward plan: **harden the solver first** (close the ~200 m RePrimAnd-vs-ours R
discrepancy to the ≲10 m level demanded by 3G/stacked measurements, using the
independent (H,β), RePrimAnd, and MUSES `qlimr` references), **then add RMFT
physics** (informed by MUSES `cmf_solver`/`crust_dft`), **then wire to RIFT**; that
is the proof-of-concept. Crust is treated as a documented systematic. Surrogates
only if acceleration is needed. Full detail in [`docs/roadmap.md`](docs/roadmap.md).


CompOSE/tabular ingestion · proper SLy/BPS crust · self-consistent RMFT Lagrangian
solver · NN surrogates (Zhu 2604.05428) · rotating NS (RNS) · gwkokab population
adapter · GP/nonparametric EOS · optional RePrimAnd and Zhu C++ δ-TOV backends.

## Free finite-temperature charged-current CPU reference

For explicitly prescribed free chemical baths, `nmbackend.reference.weak_reference`
provides detailed EC, positron-capture and neutron-decay TOTAL number/energy moments.
See [scope and conventions](docs/weak_reference.md) and
[example](examples/free_weak_reference.py). This separately imported CPU reference
does not provide an EOS solve, AD or a full weak-interaction inventory.
Selected fixed-T2 chemical-bath j/true-absorption coefficients are available only
with explicit `SpectralControls`; see [spectral validation](docs/weak_spectral_validation.md).
Finite-range reconstruction is a pilot with whole moments/tails open; no angular
transport opacity or arbitrary-bath spectral certificate is implied.
