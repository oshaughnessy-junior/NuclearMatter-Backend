# NuclearMatter-Backend (`nmbackend`)

An open-source research toolkit for neutron-star **equations of state (EOS),
stellar structure, tidal deformability, and EOS inference**. The JAX core combines
nuclear models and imported tables with an adaptive TOV solver and selected
automatic-differentiation paths. It extends the preliminary EOS tooling in RIFT's
`EOSManager.py` and provides implemented RIFT adapters and sequence I/O.

**Status: v0.0.1, research software.** Capabilities have different validation and
differentiation coverage; inclusion does not establish precision-inference
readiness. Start with the [documentation index](docs/index.rst),
[capability guide](docs/capabilities.rst), and [validation guide](docs/validation.rst).

## Capabilities

- **EOS families:** nuclear empirical-parameter metamodel with causal tail;
  nonlinear RMFT (GM1), density-dependent RMF (DD2/DDME2), standard/extended
  Skyrme (SLy4/BSkG3), polytropes, Maxwell hybrid transitions, CompOSE/simple
  tables and MUSES-CMF CSV readers, with crust and lepton/composition utilities.
- **Structure and tides:** adaptive diffrax TOV-in-enthalpy, Love numbers and
  M–R–Λ families; explicit hybrid tidal junctions and an independent radius-domain
  SciPy reference. Twin-star handling remains limited (see below).
- **Priors and inference:** constrained Brownian-bridge EOS draws, PCA sequence
  compression and Gaussian coefficient-prior tools; fixed-mass and maximum-mass
  sensitivities, Fisher forecasts, density-tagged laboratory likelihoods and
  published merger-ejecta fitting families with uncertainty estimates.
- **RIFT workflows:** `EOSConcreteNMB`, native tabular and PCA sequences, and legacy
  Landry-format export. A recorded RIFT CIP smoke test establishes plumbing,
  **not** science-grade per-EOS evidence: its 12-EOS input was remapped against an
  interpolant indexed for 10,000 EOS. See [RIFT integration](docs/rift.rst).
- **Rotation and cooling interfaces:** pinned external RNS equilibrium runner and
  tabulated-EOS adapter, external NSCool driver, and separate JAX thermal transport
  on frozen backgrounds. RNS/NSCool are not bundled; these paths do not establish
  a fully coupled rotating-star cooling model. See [rotation](docs/rotating.md)
  and [thermal limits](docs/cooling/core_transport_validation.md).

**Integration boundaries:** gwkokab is a planned adapter, despite an optional
package dependency. MUSES supplies out-of-band EOS tables/reference targets;
it is not an integrated inference dependency.

## Install and quickstart

Use a project-local environment (Python ≥3.9; core dependencies include NumPy,
SciPy, JAX, diffrax and PyYAML):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/quickstart.py
# Optional test dependencies and suite:
python -m pip install -e '.[test]'
python -m pytest
```

For the heavier validation environment and optional references, see
[installation](docs/installation.rst) and [environment notes](docs/environment.md).

```python
from nmbackend.eos import MetaModelEOS
from nmbackend.tov import build_family, m_max, radius_of_mass, lambda_of_mass
from nmbackend.adapters import EOSConcreteNMB

model = MetaModelEOS()
params = dict(K_sat=230., E_sym=32., L_sym=60., cs2_tail=0.4)
family = build_family(model.eos_table(params))
print(m_max(family), radius_of_mass(family, 1.4), lambda_of_mass(family, 1.4))
eos = EOSConcreteNMB(model, params)  # RIFT-compatible single-EOS interface
```

## Differentiation and branch limits

Use forward mode (`jax.jacfwd`, `jax.jvp`) for supported EOS/TOV sensitivities.
Reverse mode is supported for structure-only solves (`tidal=False`); reverse mode
through the tidal variable is not recommended. First derivatives in central
enthalpy are supported, but second derivatives in central enthalpy or EOS
parameters are not. Accuracy depends on tolerance, EOS representation and
transitions; the [physics guide](docs/physics.rst) records the current node/jump
handling and derivative caveats. Fixed-mass sensitivities use the implicit-function
theorem rather than coarse family interpolation; see [inference](docs/inference.rst).

`build_family`/single-EOS adapters do not yet provide complete disconnected
(twin-star) branch handling. RIFT sequence scalar accessors and legacy export
retain the primary stable rising branch; sequence branch counts let callers detect
and reject or route multi-branch EOS. They do not marginalize over all branches.

## Validation evidence and remaining work

The repository records internal comparisons against an independently implemented
radius-domain SciPy solver, including <0.1% M/R/Λ agreement for the documented
smooth/hybrid cases. Hybrid references sharing the same junction formula check
integration consistency; the smeared-transition limit separately checks the sign.
October 2026 interpolation/tidal fixes and their table-specific results are
recorded in the [physics guide](docs/physics.rst).

The initial **~200 m RePrimAnd radius mismatch is historical**: the later
[surface/interpolation report](docs/devlog/2026-06-02f-consistent-interpolation.md)
records **~3–10 m in R and <0.1% in Λ against RePrimAnd 1.7** for the compared cases.
That dated result is separate from the internal SciPy comparison and is not a
comprehensive accuracy certification of today's entire model/parameter space.
External-reference coverage, convergence, crust systematics and disconnected
branches remain work items. See the [evidence and limits](docs/validation.rst),
[validation plan](docs/validation-plan.md), and [roadmap](docs/roadmap.md).

## Scientific credit and license

MIT-licensed, by R. O'Shaughnessy and the RIT CCRG nuclear-matter group.
[ATTRIBUTION.md](ATTRIBUTION.md) distinguishes independently implemented methods,
reused data and optional external software; external licenses remain separate.
Foundations include Margueron, Hoffmann Casali & Gulminelli's metamodel
([I](https://arxiv.org/abs/1708.06894), [II](https://arxiv.org/abs/1708.06895)),
[Hinderer's tidal calculation](https://arxiv.org/abs/0711.2420),
[Damour & Nagar](https://arxiv.org/abs/0906.0096), and laboratory constraints from
[Huth et al.](https://arxiv.org/abs/2107.06229) and
[Tsang et al.](https://arxiv.org/abs/2310.11588).
