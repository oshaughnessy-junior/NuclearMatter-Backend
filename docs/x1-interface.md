# X1 interface scoping — the ejecta-conditions boundary (NuclearMatter-Backend side)

**Date:** 2026-09-05.  **Status:** scoping only — no source modules are touched by
this document or its PR.  **PR-only:** per the FRIB-SCOPING §8 decision log
(2026-09-05), merge authority covers nmb-papers papers 1–2 only, not this repo;
this lands as an open PR for RO.
**Contract source:** `n2n_paper/scoping/X1-BRIDGE.md` §4 (the authoritative spec
of what crosses the boundary), under `matter-backend-papers/scoping/FRIB-SCOPING.md`
§5 (the bridge) and its §8 decision-log entries of 2026-09-05 (both): the
differentiable ejecta-conditions surrogate ("mmapipe") lands in
NuclearMatter-Backend alongside MB-N2, PR-only; cross-repo sharing is by small
versioned data files copied with provenance headers, **no live cross-repo imports**.

**Companion schema:** [`../schemas/x1_ejecta_conditions.schema.json`](../schemas/x1_ejecta_conditions.schema.json)
(v0, JSON Schema draft-07) — the machine-readable form of §2 below.

---

## 0. Where this sits, and what already exists

The X1 bridge chain (FRIB-SCOPING §5, X1-BRIDGE §1):

```
FRIB / lab data → E_sym(n), NEPs → EOS → M(R), Λ(m)   [this repo: eos/ + tov/, AD]
                                     → ejecta conditions (M_ej, Y_e, s, τ)
                                       [this repo: the mmapipe surrogate]
                                     → n2n network adjoint → Y(A), ε̇(t)
                                       [Genesis_LANL side]
```

NuclearMatter-Backend owns the middle piece and the **downstream boundary
artifact** (§2).  The n2n side owns trajectory-condition gradients
∂Y/∂(Y_e0, s, τ) and the **upstream artifact** (§4).

**In-flight work this document builds on, and deliberately does not duplicate:**

- **PR #3, `claude/mmapipe-ejecta`** (`src/nmbackend/ejecta.py`,
  `tests/test_ejecta.py`, `scripts/ejecta_demo.py`): the differentiable
  ejecta-conditions surrogate core.  Fit families `radice2018`, `kf2020`,
  `nedora2021` in registries; a two-component (tidal/shocked) Y_e model; a
  lanthanide-weight proxy; every output a `UQ(value, sigma_fit, sigma_formula)`;
  FD check of dM_ej/dL_sym through metamodel + TOV.  Work packages WP-1 and
  parts of WP-4 in §3 are *already delivered there*; this doc only records the
  gap between that module and the boundary artifact.
- **PR #2, `claude/mb-n2-lab-likelihood`** (`src/nmbackend/lab_likelihood.py`,
  `src/nmbackend/data/frib_scenario_ladder.yaml`): the MB-N2 module and the
  provenance-header convention this artifact reuses (§2.5).

**The differentiable EOS→observable stack this composes with** lives at
`src/nmbackend/eos/metamodel.py` (`MetaModelEOS`; NEPs enter as the params dict
`K_sat, E_sym, L_sym, cs2_tail, …`) and `src/nmbackend/tov/`
(`build_family`/`family.py` interpolators, `solver.py`), giving Λ(m), M(R) and
their NEP derivatives via `jax.jacfwd` — see README "What works today" and
`docs/architecture.md`.

## 1. Design rules inherited from the contract

1. **File copies, never imports.**  Both boundary artifacts are small versioned
   JSON files with required provenance headers, copied between repos.  Nothing in
   `nmbackend` imports Genesis_LANL code and vice versa.
2. **The boundary is a pair of Jacobian tables.**  n2n supplies per-node network
   factors (Y_k, ∂Y/∂(Y_e, s, τ)|_k); this repo supplies upstream factors
   (∂(nodes, weights, M_ej)/∂NEP) and **assembles the chain-rule product** (§4).
3. **Quadrature nodes, not one linearization.**  Y_e response is the poster child
   for non-linearity (lanthanide turn-on); the artifact carries gradients at
   *multiple* Y_e nodes so composition never relies on a single global
   linearization (X1-BRIDGE §6).
4. **Honest systematics.**  NR-fit uncertainties ride along as σ's on every
   quantity (the `UQ` convention of `ejecta.py`) and propagate through the same
   linearized composition as the nuclear priors in Paper I's band.
5. **Anti-oversell.**  The symmetry-energy claim is built on the ejecta channel,
   not the mass-model channel (Giuliani et al., arXiv:2412.03243); cite it
   whenever the distinction is drawn.

## 2. Downstream artifact: `x1_ejecta_conditions` (this repo publishes)

One JSON file per scenario, validating against
`schemas/x1_ejecta_conditions.schema.json` (v0).  Contents, per X1-BRIDGE §4:

### 2.1 Y_e quadrature

Nodes {Y_e,k} and weights {w_k} discretizing the ejecta electron-fraction
distribution P(Y_e | binary, EOS).  Weights are **normalized to Σ_k w_k = 1
within each component** (mass scale carried separately by M_ej, matching the
composition formula in X1-BRIDGE §4, which scales the quadrature sum by M_ej).
Per-node (s_k, τ_k) are attached **if the surrogate resolves them**; otherwise
scenario-level `s` and `tau` objects with stated ranges are required (v0 will use
scenario-level values — see WP-2).

### 2.2 M_ej per ejecta component

Components are labeled (`dynamical`, `disk_wind`, …) since the fitting formulae
differ.  Each component carries its own mass (with σ's), optional velocity, and
its own Y_e quadrature block (see ambiguity A3 in §5: v0 is per-component; a
consumer wanting one global quadrature concatenates blocks with weights
rescaled by M_ej fractions).

### 2.3 Upstream Jacobians

∂(Y_e,k, w_k, M_ej, s_k, τ_k)/∂(NEPs) from the surrogate's AD, evaluated **at
the scenario's fiducial binary** (fixed m1, m2, χ1, χ2 — no marginalization over
binary parameters in v0; see A5).  The NEP list and its order are self-described
by a `nep_names` array (expected: `E_sym, L_sym, K_sat, cs2_tail`, i.e. the
`MetaModelEOS` params that the scenario varies).  Jacobians are stored as dense
row-per-output arrays aligned with `nep_names`.

### 2.4 Systematics

Each physical quantity carries the two-σ decomposition of `ejecta.UQ`:
`sigma_fit` (calibration papers' residual scatter) and `sigma_formula` (spread
across independent fit families), plus their quadrature sum `sigma` for
consumers that want one number (see A1).

### 2.5 Provenance header (required)

Same convention as the `frib_scenario_ladder.yaml` copy in PR #2: generating
repo + commit, generator script (path within this repo), date, upstream data
citations (which NR fitting formulae, which simulation sets), and the scenario
definition (fiducial binary + fiducial NEPs + which EOS family produced Λ, R).
A file without a complete header is invalid by schema.

## 3. Work packages (mmapipe → boundary artifact)

Candidate published fitting formulae and where they stand.  "Verified" below
means coefficient-level verification against the source PDF is recorded in the
`ejecta.py` docstring (PR #3, house one-strike rule); this scoping doc does not
independently re-verify.

| Formula | Role | Status in this repo |
|---|---|---|
| Dietrich & Ujevic 2017 (CQG 34, 105014; arXiv:1612.03665) | dynamical M_ej, v_ej functional form | Form used via the Radice et al. refit (Eq. 18 of arXiv:1809.11161); original DU17 coefficients **to-be-checked** if ever implemented as a separate family |
| Radice et al. 2018 (ApJ 869, 130; arXiv:1809.11161) | dynamical M_ej/v_ej refit; M_disk(Λ̃) tanh; Y_e component anchors (§3.1) | Implemented + PDF-verified (`radice2018`, PR #3) |
| Krüger & Foucart 2020 (PRD 101, 103002; arXiv:2002.07728) | compactness-based M_dyn (Eq. 6); M_disk (Eq. 4) | Implemented + PDF-verified (`kf2020`, PR #3) |
| Nedora et al. 2022 (CQG 39, 015008; arXiv:2011.11110) | P₂² polynomials in (q, Λ̃): M_ej, v_ej, ⟨Y_e⟩, M_disk | Implemented + PDF-verified (`nedora2021`, PR #3) |
| Sekiguchi et al. 2015 (arXiv:1502.06660) | EOS-dependence of Y_e distribution (soft vs stiff) | Cited as physics motivation (X1-BRIDGE §2); no fitting formula extracted — **to-be-checked** whether their published histograms support a quadrature shape prior |
| Lippuner & Roberts 2015 (ApJ 815, 82; arXiv:1508.03133) | lanthanide turn-on Y_e,crit ≈ 0.22–0.30 | Implemented as smooth-step proxy (PR #3) |

### WP-1  Surrogate core (fit families + σ's) — *delivered by PR #3*

Goal: differentiable M_ej, v_ej, M_disk, ⟨Y_e⟩ from (m1, m2, Λ1, Λ2, R1, R2)
with two-σ systematics.  Remaining step here: none beyond PR #3 review/merge.
Acceptance already in PR #3: FD check of dM_ej/dL_sym through metamodel + TOV
(`tests/test_ejecta.py`).  Artifact: `src/nmbackend/ejecta.py`.

### WP-2  Entropy and expansion timescale (s, τ)

Goal: attach (s, τ) to the artifact.  No implemented family provides them, so v0
ships **scenario-level values with ranges** (contract-permitted): dynamical
ejecta s ≈ 10–30 k_B/baryon-scale and τ of order ms, sourced from the same NR
literature — exact adopted numbers and citations to be fixed in the generator,
marked as assumptions in the provenance header (ladder-file discipline: every
number an assumption unless a paper commits to it).  τ follows the analytic
density-family convention of X1-BRIDGE §3 item 3 (exponential decay on timescale
τ rolling over to homologous t⁻³), so both sides mean the same τ.
Steps: (1) adopt scenario-level (s, τ) + ranges per component; (2) plumb them
through `EjectaConditions` or the generator (constant w.r.t. NEPs ⇒ zero
Jacobian rows, stated explicitly rather than omitted); (3) upgrade path: per-node
(s_k, τ_k) if a published fit resolving them is adopted later.
Acceptance: schema-valid artifact; zero rows present and labeled.
Artifacts: generator additions (WP-5); no new module required for v0.

### WP-3  Y_e quadrature construction

Goal: turn the surrogate's Y_e information (two-component mixture: tidal
Y_e ≈ 0.10 ± 0.05, shocked ≈ 0.30 ± 0.08, mass split inferred from the Nedora
⟨Y_e⟩ — PR #3) into nodes {Y_e,k} and weights {w_k} per component, with
∂(Y_e,k, w_k)/∂(inputs) available through JAX.
Steps: (1) fix a quadrature rule (Gauss–Hermite per Gaussian mixture component,
or fixed-node Y_e grid with differentiable weights — decide with the n2n side,
which owns node placement needs; nodes must bracket the lanthanide turn-on
Y_e,crit ≈ 0.26); (2) implement as a small pure-JAX function taking
`EjectaConditions` → quadrature block; (3) keep node count small (n2n cost is
one reverse solve per node × output set).
Acceptance: weights sum to 1 (per component) to 1e-12; quadrature reproduces the
mixture's ⟨Y_e⟩ and Var(Y_e) to stated tolerance; `jax.jacfwd` w.r.t. (q, Λ̃)
finite and FD-validated.
Artifacts: `src/nmbackend/ejecta_quadrature.py` (or an `ejecta.py` extension),
tests.

### WP-4  NEP Jacobians (compose with metamodel + TOV)

Goal: ∂(Y_e,k, w_k, M_ej)/∂(NEPs) at the fiducial binary, by composing
WP-3 ∘ `ejecta_conditions` ∘ (Λ(m), R(m) from `build_family(MetaModelEOS.eos_table(params))`).
PR #3 already demonstrates the composition and one FD-checked derivative
(dM_ej/dL_sym); this WP extends it to the full artifact output vector and the
full `nep_names` list.
Steps: (1) a single jitted function NEPs → artifact outputs; (2) `jax.jacfwd`
over it (NEP count ≪ output count ⇒ forward mode); (3) FD validation of **every
Jacobian row** at the fiducial point (central differences, relative tolerance
1e-4-scale, matching the repo's validation style and the X1-BRIDGE acceptance
that the surrogate's AD is FD-validated).
Acceptance: FD check passes for all rows; Jacobian finite everywhere (clamped
regimes give exact zeros per the `ejecta.py` differentiability contract, and any
clamp-active fiducial is flagged in the artifact as a warning field).
Artifacts: generator internals + tests.

### WP-5  Artifact generator + schema conformance

Goal: `scripts/export_x1_ejecta.py` — scenario definition in, schema-valid JSON
out, provenance header auto-filled (repo + `git rev-parse HEAD`, script path,
date, citation list assembled from the `FitFamily` provenance fields).
Steps: (1) generator script; (2) validate against
`schemas/x1_ejecta_conditions.schema.json` in-test (`jsonschema` dev dependency,
test-only); (3) one committed example scenario file under
`src/nmbackend/data/` for the n2n side to develop against.
Acceptance: example validates; round-trip test (generate → validate → load →
recompute one Jacobian entry) passes; provenance header complete.
Artifacts: script, schema (this PR), example JSON, tests.

## 4. Upstream artifact: the n2n trajectory-gradient table (this repo ingests)

Per X1-BRIDGE §4, Genesis_LANL publishes one JSON per network configuration:

- **Per quadrature node k:** Y(A) and ∂Y(A)/∂(Y_e0, s, τ) evaluated at
  (Y_e,k, s_k, τ_k); the same for heating functionals ε̇(t_j) on an agreed epoch
  grid (grid TBD with the n2n side — see A6).
- **Header:** network configuration (species count, A_max, rate cap R_max),
  nuclear baseline, and the run's SLURM/validation artifact paths, plus the same
  provenance-header convention as §2.5.

It arrives **by copy** into `src/nmbackend/data/` with the copy-provenance
preamble (origin repo/commit/date), exactly as `frib_scenario_ladder.yaml` was
copied in PR #2.  No code in this repo ever reaches into Genesis_LANL.

### Composition semantics

NuclearMatter-Backend assembles the chain-rule product:

```
∂Y(A)/∂NEP = Σ_k [ (∂w_k/∂NEP)·Y_k(A)
                   + w_k·( ∂Y/∂Y_e|_k · ∂Y_e,k/∂NEP
                          + ∂Y/∂s|_k  · ∂s_k/∂NEP
                          + ∂Y/∂τ|_k  · ∂τ_k/∂NEP ) ]
             · M_ej  +  (Σ_k w_k Y_k(A)) · ∂M_ej/∂NEP
```

(the s, τ terms vanish in v0 where s, τ are scenario-level constants).
**Validation split:** each side validates its own factors independently — n2n
against finite differences on the network, this repo against WP-4's FD checks —
and the assembled chain gets **one end-to-end finite-difference spot check on a
single NEP direction** (perturb e.g. L_sym by δ, regenerate the downstream
artifact, request one re-run from the n2n side at the shifted nodes, compare
against the assembled linear prediction).  That spot check is the only step
requiring a coordinated two-repo exchange, and it is still file-mediated.

## 5. Contract ambiguities resolved here (to reconcile with the n2n side)

- **A1 — one σ or two.**  X1-BRIDGE §4 says systematics are "carried as σ's"
  without decomposition; `ejecta.UQ` carries (sigma_fit, sigma_formula).  The
  schema carries both **plus** the combined `sigma` (quadrature sum), so either
  reading works.  n2n should confirm which it consumes.
- **A2 — per-node (s_k, τ_k) unresolved in v0.**  The contract's "if the
  surrogate resolves them, else scenario-level with ranges" clause is exercised
  on its *else* branch: no implemented fit family resolves s or τ.  v0 artifacts
  are scenario-level; the schema supports both forms.
- **A3 — quadrature scope.**  X1-BRIDGE §4 lists *a* Y_e quadrature and M_ej
  *per component* without saying whether the quadrature is global or
  per-component.  Since the fits and Y_e physics differ per component, v0 makes
  the quadrature **per-component** (weights summing to 1 within a component,
  M_ej as the separate scale factor, matching the composition formula).  A
  consumer wanting a global quadrature concatenates with weights scaled by
  M_ej-fractions.
- **A4 — NEP list and order.**  Not fixed by the contract; the artifact is
  self-describing via `nep_names` (expected `E_sym, L_sym, K_sat, cs2_tail` from
  `MetaModelEOS`).  The n2n side should index Jacobians by name, never position.
- **A5 — fiducial binary, no binary marginalization.**  Jacobians are at fixed
  (m1, m2, χ1, χ2) per the contract's "at the scenario's fiducial binary".
  Marginalizing over a binary posterior is a later version (schema has room via
  new scenario files, not new fields).
- **A6 — heating epoch grid.**  The upstream table's ε̇(t_j) epoch grid is
  "agreed" but nowhere fixed.  Proposal: logarithmic grid 0.1–30 days
  (kilonova-relevant), ~8 epochs; to be confirmed by the n2n side and then
  recorded in both schemas.
- **A7 — τ definition.**  Adopted: the analytic-family convention of X1-BRIDGE
  §3 item 3 (density e-folding timescale of the exponential phase before
  homologous rollover), since that is what the n2n μ-equation traces.

## 6. Non-goals

- No source-module changes in this PR (scoping doc + schema only).
- No duplication of PR #3 (surrogate) or PR #2 (MB-N2): WP-1 is explicitly
  "delivered elsewhere"; WP-2–5 are the gap.
- No live cross-repo imports, in either direction, ever (FRIB-SCOPING §8).
- No astrophysical claims: the 411-species demonstrations downstream carry
  Y > 0 only to A = 112; peak-region claims wait on the full-network run
  (X1-BRIDGE §6) — nothing in this interface changes that.
