# Dev log — 2026-06-02 (d) — RePrimAnd working; forward roadmap

**Who:** R. O'Shaughnessy (with AI pair).

## State
- RePrimAnd 1.7 cross-check now runs (keyword-only `star_acc_simple` fixed in both
  RIFT `EOSManager` and nmbackend `reference/reprimand.py`).
- **Science finding:** RePrimAnd vs our solver disagree by **~200 m in R** in some
  cases (~1-2%), vs our internal diffrax-vs-SciPy(H,β) agreement of ~2 m.  This is
  the gap to close for precision EOS inference (3G/stacked → tens-of-meters
  statistical precision → numerics budget ≲10 m in R, ≲0.5% in Λ).

## Plan (docs/roadmap.md), per RO's sequencing
1. **Harden solver first** — we already have adversarial references (independent
   (H,β), RePrimAnd, MUSES `qlimr`).  Diagnose the 200 m: prime suspects are the
   log-linear EOS interpolation + cs² clipping/`cummax` smearing and the surface
   convention (we stop at table `h_min`, RePrimAnd at its own surface).  Build a
   reference-matrix harness with per-quantity error tables; convergence study; full
   adversarial EOS ladder incl. twin stars; add I(M)/I-Love-Q.
2. **Add RMFT physics** — self-consistent σ-ω-ρ solver under the same EOS interface;
   keep the NEP metamodel as the fast model.  Use MUSES `cmf_solver` (Chiral Mean
   Field) as reference EOS generator + validation, `eos_taylor_4d` for the metamodel,
   `njl` for quark/hybrid; CompOSE ingestion.
3. **Wire to RIFT** — EOSSequenceLandry HDF5, `util_RIFT_pseudo_pipe` tabular path,
   then gwkokab + GP.

## MUSES integration note (link added)
MUSES (`2026-04-22-Explore-MUSES/`) is a containerized Django/Celery stack
(PostgreSQL/RabbitMQ/Celery/MinIO), brought up via Docker (API :4000).  We use it
**out of band** — trigger module workflows via the API and mount the resulting
CompOSE tables — as an EOS provider/reference, never in the hot path.  Relevant
modules: `cmf_solver` (RMFT), `crust_dft` (crust), `eos_taylor_4d` (metamodel),
`njl` (quark), `qlimr` (M-R-Λ-I observables).

## Crust
Treated as a documented systematic, not a solved problem: prefer a unified
crust-core table (MUSES `crust_dft`) else SLy/BPS matched at a transition; quantify
the crust-induced R/Λ spread; consider marginalizing over crust choice for inference.

## Milestone
Phases 1-3 = proof of concept.  Surrogates (Zhu 2604.05428/2604.24949) + RNS only if
the hardened solver is too slow in the loop.
