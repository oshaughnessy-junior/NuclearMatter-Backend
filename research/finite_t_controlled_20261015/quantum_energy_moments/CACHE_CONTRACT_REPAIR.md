# Exact cache provenance repair

2026-10-15. The original `run_study.py`, both unqualified-provenance partial reports, note and checkpoint are preserved under `pre_cache_contract_20261015T1332/` with SHA256 manifest. Existing reviewed physics results are unchanged. This is a cache-path repair, not new quadrature evidence.

Fresh runs remain the default. Explicit `--resume` now validates the entire cache before returning any hit. Each record carries its complete fingerprint and SHA256: full solved nested FD state plus explicit T, nn, np, kinetic mass and species offsets; all nine local transitive physics-source hashes; hashes of the integrator, driver and cache-contract implementation; Python/executable, NumPy, SciPy and platform versions; flavor, outer order, internal order, internal tail, slab length and every energy-panel edge. Exact JSON serialization rejects nonfinite values. The result itself has a digest and must match the requested controls and panel boundaries. Duplicate, unknown, stale, malformed or missing provenance is rejected. A missing requested resume file is an error. Old partial files are deliberately excluded; they cannot be silently migrated from tag/flavor/order alone.

This fingerprints local physical source files and records installed numerical-package versions; it is not a binary reproducible-build certificate for NumPy/SciPy. It also does not establish that maliciously rewritten data are truthful.

`check_cache_contract.py` performs one actual tiny physical quadrature, verifies exact replay, changes each state/control/source/environment category, corrupts result data, and tests missing/legacy/duplicate records. It also runs the actual entry point against the existing legacy partial, verifies early rejection and verifies that its bytes remain unchanged. **31/31 checks pass in 1.97 s.** Full 602-gate physics quadratures were not rerun because the kernel and its numerical path were not modified. The root legacy partials remain present for the reproducible rejection test; later fresh study runs will replace their working-path copies, while preserved originals remain in the snapshot directory.

The source’s positive slab pieces are **linear transfer components**, not gross microscopic emission j(1−f) and absorption χf counts. Relaxation attenuation includes emission blocking. Existing total number/energy/lepton ledgers remain the reviewed quantities.

Independent cache recheck is required before acceptance.
