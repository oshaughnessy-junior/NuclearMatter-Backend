# Validation deepening (code thrust)

Tracks the code-improvement thrust running in parallel with the paper sequence
(papers live in the sibling `matter-backend-papers/` repo). Focus: more independent
validation, so the methods paper rests on cross-checked numbers.

## RePrimAnd (run on the pyreprimand machine — not this sandbox)

- Current: `scripts/validate_against_reprimand.py` exists; M/R agree to ~3-10 m with
  the matched surface convention. Run the full sweep on the machine that has
  `pyreprimand` installed and archive the comparison table.
- **Gap:** the independent **tidal-Λ** cross-check is thinner than M/R. RePrimAnd
  provides Λ — add a dedicated Λ-vs-RePrimAnd comparison across the metamodel and
  RMFT families and a couple of phase-transition EOS. (One afternoon; paper figure.)

## MUSES / CMF (here)

- Current: `docs/muses-integration.md`; CSV ingestion via `load_cmf`; couplings
  documented. Model-level (ε, p, μ, cs²) diagnostic exists.
- **Next:** drive the MUSES `cmf_solver` (see `2026-04-22-Explore-MUSES/`) to generate
  a reference CMF table, ingest it, and run the model diagnostic + a TOV cross-check
  (M-R-Λ) against an independent solve of the same table. Confirms our ingestion +
  TOV on an externally-authored, multi-author-vetted EOS.

## Throughput for paper-scale ensembles

- **10⁴-draw bridge ensemble** → full-cardinality RIFT tabular run (closes the
  smoke-test cardinality gap). The bridge prior generation is cheap; the TOV family
  builds are the cost. Vectorize/vmap the family build over draws, or batch on the
  cluster. Target: one overnight run producing `bridge_prior.nss.h5` (+ PCA) at 10⁴.

## Adaptive-prior substrate (feeds paper 1 N1/N2)

- New module: GP (or empirical-interpolant) prior over the **PCA coefficient space**
  of the compressed representation (`pca_hc/1`). This is the adaptive substrate; the
  iterative refinement loop lives with RIFT-GP. Keep the *representation* and
  *sensitivity* code here; keep the *algorithm/paper* in the papers repo.

## Standing validation rule

No new physics model (RMF set, NJL, hyperonic, ...) is exposed without reproducing
its published saturation point / an independent reference solve (RePrimAnd, O2scl,
MUSES). NL3 lesson.
