# Independent review: integrated-moment resume cache contract

15 October 2026. Author confirmed freeze before testing. Pinned SHA-256:

- `cache_contract.py`: **1f40edb4887bfed2052e241f05fc840ea9eb9f78831d5e9e3766c94360357460**
- `run_study.py`: **faf527020caf55a7efae542ecf8eedeb070d02ffdd96bc21d48e6b198b7f9fd5**

Reviewer changed only new `review_physics/cache_repair_*` files. No author cache/result was altered, no full physics integration was replayed, and no dependency installed. Account/window matched at start (7% used).

**Accept the repaired resume contract for the frozen-source workflow.** The previously identified flavor/order-only compatibility issue is closed. This acceptance does not change any physics qualification: the moderate coarse-grid failures remain failed, their accepted replacement is the separately reviewed refinement, and the heavy-M complete-inventory gate remains open.

The fingerprint includes the complete FD matter dataclasses and explicit source state inputs, mass/rest offsets, nine upstream physics-source hashes, driver/integrator/cache code hashes, Python/executable/library/platform versions, flavor, outer and inner orders, inner tail, slab length and complete energy-edge list. Boundary functions and physical constants are covered by their source hashes. Floating-point/nonfinite provenance is rejected during canonical digest construction.

Each record contains the exact fingerprint plus its digest and the result plus its separate digest. Result flavor/orders/tail/length and panel edges must match the request. Loading compares every record with the complete current expected request set, rejects unknown/stale/duplicate/corrupt/legacy records, and returns hits **only after the whole file passes**. A valid prefix followed by a stale record cannot supply a partial accepted cache. Missing files are rejected under explicit resume.

The fresh default does not load a cache: `cache.load(...)` is selected only with `--resume`; otherwise the existing integration path computes fresh results and records exact provenance. No physical kernel, coefficient normalization, energy quadrature formula or scientific tolerance is changed by this protocol repair. Old partials are deliberately incompatible with the new schema; their scientific evidence is preserved rather than silently blessed with newly generated provenance.

`cache_repair_probes.py` uses only temporary **synthetic** payloads. All **8 independent checks pass**, saved in `cache_repair_probe_results.json`:

1. Exact compatible synthetic replay.
2. A cache generated for a genuinely different EOS temperature, with its metadata internally consistent, is rejected against the requested state.
3. Mutating an upstream source hash and **recomputing** the metadata digest still fails the current expected fingerprint comparison.
4. A valid leading record followed by that stale record rejects the entire file.
5. Duplicate records reject.
6. Result corruption rejects.
7. Legacy metadata-free data reject.
8. A missing resume file rejects.

Source SHA values are checked before and after these probes. The test does not rewrite real source files to simulate upstream changes, and it does not edit or reclassify any author result. It complements the author's 31 checks, which include a tiny actual integration/replay and actual entrypoint rejection.

This is a scientific compatibility/integrity protocol, not a new physical validation or a guarantee for changing source files underneath an already imported running process. Continue the campaign's frozen-source discipline. Before reusing a cache under different physics or controls, construct current fingerprints and let the loader reject it; do not retrofit old fingerprints onto unverified payloads. The publication interpretation of positive slab transfer pieces and all limits in `energy_moments_review.md` remain unchanged.
