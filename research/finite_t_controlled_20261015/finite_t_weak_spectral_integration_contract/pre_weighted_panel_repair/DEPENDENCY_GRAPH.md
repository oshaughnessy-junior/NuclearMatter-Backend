# Stage B dependency graph and immutable boundaries

Proposed future installed graph; no implementation/import/physical evaluation performed here:

```
weak_reference.spectral_coefficients / capture_differential_coefficients
   -> _weak_spectral_common (new spectral-only support/positive-log/result types)
   -> _weak_spectral_capture (new conditional capture geometry)
          -> _weak_current.scalar_capture / literal primitives (Stage A, read-only)
          -> _weak_quadrature.log_fd (Stage A, read-only)
   -> _weak_spectral_decay (new lab-energy conditional parent/cone geometry)
          -> _weak_current.decay_amplitude / boost (Stage A, read-only)
          -> _weak_quadrature.log_fd (Stage A, read-only)

weak_reference.reconstruct_spectral_moments
   -> _weak_spectral_common bounded incident-energy quadrature/DOS/actual f and holes
   -> actual new spectral_coefficients at each node, no saved-rate lookup

independent acceptance comparison
   -> Stage A channel_moments event integrals (separate original phase)
   -> independently authored shell/spinor/conditional-delta references in tests
```

The proposed first Stage B does not change Stage A's `_weak_quadrature.py` strict PositiveQuantity zero semantics. It adds a separate spectral-only exact-kinematic-support record. It does not import `_weak_decay.rest_events` as a conditional-spectrum substitute: the total rest lepton-pair invariant and conditional electron–proton pair invariant are different. Existing moment/EOS/thermal research drivers do not become dependencies.

Current research loaders being replaced:

- hot EC spectrum dynamically loads leptons/EOS/moments/vacuum/current/tensor, verifies campaign files and restores selected generic aliases. Its `matched()` runs fresh EOS and exact-state comparison. Portable Stage B receives the explicit supplied chemical bath and does not reproduce this hidden EOS/loader chain.
- hot PC spectrum shares that loader/geometry but calls a scalar amplitude currently under `review_pc_moments`. This reviewer directory remains an independent evidence oracle, never installed runtime. The Stage A source-derived physical scalar/current provider is the runtime primitive.
- hot beta spectrum imports generic beta/kernel/tensor/current/vacuum/free_baryons/events through restored aliases, plus a frozen result SHA, and runs `matched_npe`. The portable conditional phase must be extracted mathematically, retaining direct occupations/support/absolute spin normalization without source-file IO, old low-T guards or dynamic loaders.
- current reconstruction consumers read node files and qualification registries. The first installed reconstruction uses fresh in-memory direct quadrature, frozen state/model/control objects and explicit complete status. Any later checkpoint/cache protocol is separately reviewed and cannot turn a partial collection into a physicalzero.

No base package `__init__`/cold API changes; new helpers use package-relative imports and already declared NumPy/SciPy only. Actual minimum-runtime API compatibility, real installed origins, no driver side effects and package wheel contents are future independent gates. The repaired Stage A candidate is pinned as a prerequisite with DIFFERENT selected numerical/physics acceptance; packaging/CI/minimum-runtime/merge remain separate. It is not adopted as an implicit unmerged dependency, and that narrow acceptance does not authorize or qualify Stage B.
