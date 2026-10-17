# Independent repaired slab-input recheck

**Accept the scoped repaired input contract at source13c3bee74aa03803693416082271b5fad2dba5d6e8e1759e9eb2020be6694dfa:74/74 selected recheck gates PASS.** The original review's46 physical passes and three input failures remain byte-identical, with its source/raw/note/manifest snapshot intact. This is a separate repaired-source result; the original failures were not relabelled as passes.

The exact old-source bytes still reproduce nonfinite `rhs` outputs for NaN left boundary and NaN cell populations with collisions disabled, plus a successfully evolved negative-temperature reservoir. The new source rejects each case with ValueError. New independent probes extend these to both boundaries, collision ON/OFF, infinities, zero/unit populations, complex/string/Boolean arrays, wrong shapes, zero cells, nonliteral flags, nonfinite time controls and nonpositive thermal temperatures. Invalid evolution controls are tested with a forbidden ODE-entry sentinel, so rejection must occur before the solver starts. Selected invalid E/W/A/angular-node inputs also reject. No source patch is made by the reviewer.

A fresh valid anisotropic two-cell state has bitwise-identical boundary, collision, collision-OFF/ON RHS and face/thermostat ledger outputs between exact old and repaired source. A NEW complete trajectory (duration.011s, crossing.043s, initialTr1.93MeV, left/rightTr2.4/1.7MeV, mu_nu−.52MeV) likewise agrees bitwise across every returned field except wall-clock runtime. This compares actual recomputed nonlinear populations, not just saved outputs.

AST comparisons establish `angular_edge` and `grid` are unchanged in full, and collision, upwind face/ledger and ODE arithmetic are unchanged after guard/setup code. All four elementary physical source hashes match the original reviewed pins. Thus the earlier independent relative-azimuth/target-energy quadratures and Radau finite-grid physics checks continue to apply to unchanged arithmetic; no full expensive grid rerun is needed from this reviewer.

The accepted contract remains deliberately strict: exactly shaped finite real interior occupation arrays and positive finite FD thermal reservoirs. Saturation/underflow and malformed grids reject rather than being clipped or floored. This is not a complete arbitrary-grid/untrusted-input API certificate.

Three coarse azimuth failures and the2.79875% three-to-six-cell heat sensitivity remain retained/excluded. The finite S2/four-energy-node/24MeV ansatz, thermostat treatment and open spatial boundary ledgers are unchanged. No continuum angular/energy/spatial accuracy, open-system entropy theorem, evolved material or hydro claim is added. See the initial `REVIEW.md` for the physical qualification and its boundaries.

Evidence: `repair_recheck.py`, `repair_recheck_results.json`, and `REPAIR_RECHECK_MANIFEST.json`. The initial `MANIFEST.json`, initial probes/results and all other original review files are unchanged.
