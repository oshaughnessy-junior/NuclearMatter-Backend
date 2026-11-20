# Independent repaired-helper recheck

**ACCEPT the scoped helper-admission repair at sourceSHA256 `d8400295e4bc286ed5453c6da70d5f3d1cfc76e94f926b3eb7603e0ea55d7d14`.82/82 independent rechecks PASS in0.041s.** This does not change the physical acceptance boundary: the finite-Q external-leg vector defect remains nonzero, interaction currents are absent, and no complete modified-Urca current/rate/width is supplied.

The initial review remains intact:419 passing finite-input/algebra checks plus TWO failing helper controls. The earlier404-pass checkpoint preceded the final15 Slater-normalization checks. No initial source, report, manifest or failure has been overwritten. Their exact hashes are recorded before/after in `repair_recheck_results.json`.

The new script loads **only** the preserved original pure function source and repaired pure function source under different explicit module names. It never imports the initial review probe, so the initial reports are not replayed or rewritten. The original file independently matches the author's `pre_helper_repair/external_legs.py` bytes and the reviewed372ae513…hash; the new file matches the commissioned d8400295…hash. The separately reviewed strong OPE source also matches its original hash before/after.

The exact two original counterexamples are reproduced against the preserved old code: `gv=NaN` returns a nonfinite matrix, and `raising=NaN` returns the identical finite matrix to `raising=True`. Against the repaired helper, both now raise ValueError. Additional NaN/infinity/Boolean/complex/string/None coupling and non-Boolean raising probes reject. Valid Python and NumPy Boolean charge choices and finite real couplings remain accepted.

At both original independently derived heldout finite-Q points, all7 amplitudes for lowering/raising weak charge and the denominator-free helper are **bitwise unchanged** between old and repaired code for finite gV=.83,gA1.17. The independently assembled finite-Q vector contraction still matches the helper's antisymmetrized commutator at the original3×10⁻¹² relative gate. No current-conservation zero or weak matrix-element accuracy is inferred from this guard repair.

A structural AST comparison verifies unchanged `validate`, `sigma`, `amplitude`, `channel` and `emission_point` functions exactly. In `rho`, `direct` and `vector_defect_direct`, removing only the explicit argument guards/real-scalar normalization leaves identical numerical bodies. The added finite-real coupling validator and Boolean raising guard account for the patch; no transfer, energy denominator, Pauli/isospin sign, normalization factor or physical current formula changes.

All repaired author source/raw artifacts are unchanged during this independent recheck. Initial failure evidence is preserved byte-for-byte. This accepts the stated repair of those helper controls, without a broad generic API, physical model, finite-T or rate certification.
