# Scoped finite-input repair after the independent slab review

**Repaired author freeze; a different independent repaired-source recheck is required.** The independent reviewer accepted46 finite-input physical/algebra checks but rejected three input cases at initial source `be7f2c05f595647ef9fa7a868bd9b8b4d78f39dee4b2b2055c32180334906022`: NaN boundary and NaN cell populations propagated with collisions off, and a negative-temperature incoming FD reservoir was accepted. Those are control defects, not new physical slab states. Their raw independent failures remain unchanged in the reviewer-owned directory.

Before editing, all15 initial top-level author files were copied byte-for-byte to `initial_input_failure_snapshot/`, including source, initial and fine physical reports, checkpoint, manifest, notes and figure data/artifacts. SNAPSHOT_PINS.json records exact original hashes. The independent initial reviewer separately preserves its source/raw/note/manifest and INITIAL_PIN; this repair does not edit any reviewer or upstream files.

## Guard-only changes

- Finite real numeric arrays are validated explicitly before comparison/arithmetic, so NaN no longer slips through the `<=0`/`>=1` population comparisons. Entire Boolean, complex, string or object arrays are rejected rather than coerced into a physical population array.
- Cell state and left/right boundary arrays must have the exact expected shapes and strictly interior finite occupations0<f<1, including collision-OFF calls. Zero cells, flattened arrays and broadcastable wrong boundary shapes reject.
- The physical grid requires finite positive E/W, matching real interior angular nodes, finite nonnegative A with positive off-diagonal transitions, and matching shapes. Self-mode diagonal activities are still ignored by the unchanged paired equations; finite alternate diagonals do not change the physical cancellation.
- Boundary construction requires positive finite left/right temperatures, finite chemical potential and positive finite energy nodes. Underflow/saturated radiation occupations reject instead of being floored. Evolution also requires positive finite initial temperature and rtol, finite bounded time controls, and a literal Boolean collision switch. Invalid thermal and scalar controls reject before the ODE solver is invoked.

No physical kernel, transition direction, DOS/spin factor, source normalization, coherent subtraction, streaming formula, boundary flux, thermostat sign, entropy expression, time algorithm or numerical tolerance is changed. Array validation adds admission checks only. The original finite-energy/spatial/angular limitations remain in force.

## Meaningful regression and physical replay

`check_repair.py` has **116/116 passes** in.63s. It reproduces the three original input failures, extends them to both boundaries, nonfinite values, shapes, zero/out-of-range occupations, numerical controls, literal flags and invalid physical arrays. For the invalid thermal/control cases it replaces the ODE entry with a sentinel that raises if reached, proving these errors reject before solver entry. It executes the exact preserved original source at its original module path solely to resolve the same frozen function dependencies; no old driver is imported or run.

Valid boundary values, collision arrays/scalars, collision-on and collision-off RHS/ledgers, and a complete fresh valid two-cell trajectory are **bitwise identical** to the preserved source (wall-clock runtime excluded). AST comparisons show angular_edge/grid unchanged; exact source-tail comparisons show collision, face-flux/RHS and evolution arithmetic unchanged after their validation setup.

Both original physical batches were actually rerun under the repaired source:

```
python check_repair.py
python check_slab.py
python refine.py
python audit_replay.py
```

The original batch repeats32PASS/3coarseFAIL in5.19s; the refined batch repeats23/23PASS in18.75s. The three coarse azimuth failures remain exactly the same, with unchanged3e-6 gates. `audit_replay.py` adds6/6 passes: complete initial/fine/checkpoint raw JSON calculations are exactly equal to the preserved original after removing only `seconds` fields; every original snapshot file and all four frozen physical source files still have their original hashes. Thus no new source calibration or altered physical answer is hidden behind the guard repair.

Current thermostat heat remains2.948230139138167e-8MeVfm^-3, and three→six-cell heat sensitivity remains2.7987517469%. The original four-energy-node S₂/24MeV/three-cell scope and missing continuum gates are unchanged. Figure bytes need no regeneration because their plotted physical data are identical.

One initial repair-validator attempt had a missing whitespace SyntaxError and produced no result; exact failed driver/error are preserved as `initial_repair_driver_*`. This was confined to new test code. No assertion or physical source was changed to hide it.

Author accounting:55 passing original physical assertions,116 guard/bitwise checks and6 exact raw-replay audits, with three original coarse quadrature failures retained. Counts are related controls, not177 independent physical benchmarks. The new source is frozen for independent recheck; no repaired acceptance, generic untrusted-code protection, full API domain, continuum transport or production status is claimed.
