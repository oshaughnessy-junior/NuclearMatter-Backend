# Overnight status — EOS backend (post RIFT-integration smoke test)

## Session 2 addendum (own prior + docs)

Per direction ("generate our own prior… Gorda-style GP priors with QCD consistency"
and the pointer to arXiv:2512.18044):

1. **Constrained Brownian-bridge prior** (`nmbackend.priors.bridge`,
   `scripts/draw_bridge_prior.py`) — full implementation of
   Gorda-Komoltsev-Kurkela-Sunde 2025: KK 3-volume (verified against the
   2111.05350 formulas and Table I anchors), exact uniform sampling in the
   (μ,n,p) volume, self-similar fractal refinement, diffusion smoothing.
   One deliberate reformulation: the diffusion is done as constant-D smoothing of
   ln μ vs ln n — equivalent to the paper's D∝n² (σ/n = const) but with an exact
   maximum principle for cs², after the paper's μ(n) form produced cs² up to ~8
   under a tapered D (taper violates their App. C condition). Result: cs² ∈ [0,1]
   exactly, μ/p monotone, dε/dn=μ to machine precision, μ anchors exact, pQCD
   pressure endpoint within a few % (inside the X band). Draws make physical stars
   (M_max 1.5–1.9, R(1.4) ≈ 11–12 km). 5 tests in `tests/test_bridge_prior.py`.
   Low anchor = our metamodel band (NEPs sampled per draw) → the prior is OURS.
2. **LCEHL recovery** (`scripts/convert_landry_to_nss.py`) — repackage the
   published files into NSS (verified on the real 10⁴-EOS PSR+GW+NICER file), or
   re-solve their microphysical tables with our TOV (`--mode resolve`).
3. **Sphinx/readthedocs site** (`docs/` + `.readthedocs.yaml`) — user guide
   (installation/quickstart/physics/priors/representations/RIFT/inference), API
   autodoc with mocked heavy deps, design notes + devlog in toctrees; builds clean
   (21 pages) with `sphinx-build`.

Commits: `1f1258d` (bridge prior), `c495d63` (LCEHL converter), `d1dc4fe` (docs).


Picking up after `SMOKE TEST PASS`. Three deliverables landed, each committed with
tests; the full suite is green (every test file run, no regressions). Summary below.

## 1. Compressed / functional sequence representation (the "not just draws" mandate)

A real **non-tabular** `SequenceRepresentation`: per-channel PCA/SVD over the
M(u), R(u), logΛ(u) curves → `mean + components + coeffs` (`representation="pca_hc/1"`).

- `nmbackend.rift.compressed`: `fit_sequence_pca`, `write_pca`, `write_pca_from_nss`
  (compress an existing `.nss.h5` with no TOV re-run), `PCASequence` reader.
- `open_sequence` dispatches on the tag; RIFT `EOSSequenceFromFile` now routes
  pca / tabular / legacy → `EOSSequencePCA` / `EOSSequenceNMB` / `EOSSequenceLandry`.
- `scripts/compress_sequence.py` CLI; `tests/test_rift_compressed.py`.
- **Accuracy:** at k=5–6 components, Λ(1.4) reconstructs to ~0.1% (max 0.2%), R≤3 m,
  M_max≤0.5 mM⊙. Compression grows with the draw count (≈12× at 10⁴ EOS).
- This is the seam for GP / empirical-interpolant priors later: any coefficient vector
  reconstructs a smooth, valid M-R-Λ sequence — a function-space model, not a replay.

## 2. Fisher / sensitivity utility (the differentiability payoff)

`nmbackend.inference.fisher`: `sensitivity_matrix`, `fisher_matrix`,
`fisher_covariance`, plus `examples/fisher_forecast/`.

- **Found and fixed the real gradient bug.** The flagged "Λ-gradient needs rtol≲1e-12"
  problem was a red herring: the catastrophic error (dΛ/dcs2_tail came out *wrong
  sign*) was from differentiating through the **family interpolation at fixed mass**,
  not the solver. Fixed-mass sensitivities now use the **implicit-function theorem**
  (M_max via the **envelope theorem**), from accurate single-star tangents.
- **Validated against the correct reference** — a fixed-mass FD that re-roots h_c
  (Newton-polished) per perturbed EOS: **dΛ/dcs2_tail 0.0%, dΛ/dL_sym 2.5%, dR ~0.3%,
  dM_max ~0.1%**. The earlier 13%/137% "errors" were noise in the family-interp FD
  reference, not the method.
- Net: the gradient-tightening backlog item is **resolved** for Fisher/HMC use; the
  n-μ core-EOS rewrite is **not needed** (I deliberately did not touch the validated
  value path for a ~2.5% residual). See `docs/backlog.md` and `tests/test_fisher.py`.

## 3. Live CIP run with our backend (from the prior session, recorded)

`examples/gw170817_tabular/SMOKE_TEST_RESULT.md`: CIP reproduced the reference
(lnL 463 vs 464.7) on the legacy path, and ran end-to-end on our `EOSSequenceNMB`
(lnL 458) — the integration is real. The PCA path now also drops in via the same
factory.

## Commits

NuclearMatter-Backend: `241d9ee` (PCA), `6e08aa0` (Fisher/IFT), `b0a71db` (forecast
example), `e1d1d40` (re-rooted FD validation + backlog). RIFT: `e27b92f0`
(EOSSequencePCA + 3-way dispatch).

## Recommended next (for when you're back)

1. **Science-grade tabular run.** The only thing the live CIP test couldn't do
   honestly is a same-cardinality sequence — `all_tabular.net` is bound to the 10⁴-EOS
   LCEHL indexing. Generating ~10⁴ nmbackend EOS is a throughput task (vmap the family
   build); worth doing to get a real per-EOS evidence curve to compare against LCEHL.
2. **Break the L_sym/K_sym degeneracy** the forecast surfaced — add Λ at multiple
   masses (1.2/1.6 M⊙) and a second radius; the Fisher tool quantifies the gain in
   seconds.
3. **GP prior over PCA coefficients** — the compressed rep is ready for it; this is the
   path to sampling new EOS between draws rather than replaying a fixed bank.
4. Additional RMF sets still need verified couplings before exposure (NL3 lesson);
   best done with a reference table in hand, not unattended.

## Open question for you

For the science-grade run (item 1): do you want the nmbackend sequence drawn to *match*
LCEHL's prior (so the evidence is directly comparable), or drawn from our own
metamodel/RMFT prior (a genuinely independent cross-check)? That choice drives how I
generate the 10⁴-EOS file.
