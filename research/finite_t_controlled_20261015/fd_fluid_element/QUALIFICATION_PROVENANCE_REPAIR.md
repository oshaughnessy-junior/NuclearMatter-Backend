# Initial physical-source qualification provenance repair

2026-10-15. Original files/results are preserved under `pre_qualification_contract_20261015T1402/` with SHA256 manifest. The physical EOS, rate kernels and source/step integrators are unchanged; only qualification stamping/loading and rejection paths changed.

An initial physical source may now be reused only when its payload passes and matches an exact fingerprint of input T,nB,Ye, photonsFalse; the entire solved nested FD state and thermodynamic closure; all nine local transitive physical source hashes plus element/source integrator, qualifier, trajectory driver and contract hashes; Python/executable, NumPy,SciPy and platform versions; coarse/fine outer orders, internal order/tail, every energy edge, complete channel inventory and empty-radiation helicity convention. A payload checksum prevents silently changed moment data. The actual source metadata is separately compared with the requested state/controls. Legacy/missing/stale provenance is rejected explicitly before reuse.

`qualify_source.py` now captures the fingerprint before its physical calls, verifies it again afterwards, then stamps the result. It was freshly run once:24/24 physical/EOS gates still pass in37.66s, with the same initial rates and derivatives. `short_evolution.py` validates the current fingerprint before loading that shared exact initial state. No trajectory was rerun because its physical integration path is unchanged; historical short-step results remain preserved and numerically accepted at their prior scope.

The pure contract regression changes every state/control/source/environment category, corrupts/re-hashes wrong data, and excludes the preserved legacy payload. Its results and source hashes accompany this checkpoint. Installed numerical-package versions are recorded; this is not a full binary reproducible-build certificate.

Independent provenance recheck is required. This repair does not qualify longer trajectories, the failed heavy-static transfer wrapper, or a broader physical EOS/rate model.
