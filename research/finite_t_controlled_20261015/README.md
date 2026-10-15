# Finite temperature references for controlled transport

Research-only CPU calculations from the user-authorized 15 October 2026 campaign. No production `nmbackend` API, dependency pin or existing cooling implementation changes. These references support simplified fluid elements, fixed-background slabs and local uniform mean-field states; a full hydro solver and production GRMHD throughput are outside scope.

## Reproduce

Use the repository's existing NumPy/SciPy environment. From the repository root:

```
python research/finite_t_controlled_20261015/thermodynamics/validate_reference.py
python research/finite_t_controlled_20261015/weak_rates/check_reference.py
python research/finite_t_controlled_20261015/controlled_models/slab_benchmark.py
python research/finite_t_controlled_20261015/spectral_slab/run_slab.py
python research/finite_t_controlled_20261015/fluid_element/fixed_temperature_element.py
python research/finite_t_controlled_20261015/thermal_element/cooling_element.py
python research/finite_t_controlled_20261015/rmf_reference/validate_gm1.py
```

Scripts write measurement JSON beside their sources. Initial evidence and source hashes are preserved by this archive; timestamps/environment metadata change on a rerun. No new packages are required, and the figure generator is intentionally omitted from this backend code slice. Figures and scientific interpretation live in the [papers repository](https://github.com/oshaughnessy-junior/nmb-papers/blob/master/notes/RESULTS-FINITE-T-REFERENCE-SLICES-2026-10-15.md).

## Scope and review status

The ideal lepton calculation passes 86 checks after an independently discovered tiny-charge inversion defect was repaired; the static weak calculation passes 44 checks. The explicit classical common-kinetic-mass nucleon closure keeps the physical neutron-proton Q difference but omits recoil, Pauli blocking and mean fields. It is appropriate only as the named controlled ansatz, not a dense-core Urca rate.

The fixed-bath spectral slab passes 40 checks and independent opaque-limit normalization. Its warm positive-chemical-potential example is accepted; generic strongly negative chemical potentials and cold narrow Fermi edges need new energy-grid qualification. The static antineutrino operator has disconnected 0.7823–1.8043 MeV modes: arbitrary trapped spectra cannot relax globally without an additional redistribution mechanism or invariant-aware acceptance condition.

The thermostatted composition element and no-thermostat isochoric cooling element have separate independent physics reviews under `review_physics/`. The latter passes six trajectory runs, independent energy/charge Jacobian checks and endpoint energy integration. Thermostat heat must never be confused with an internal cooling source.

The GM1 local uniform reference passes 197 author checks and 31 independent numerical checks. Its archived cold NMB sources match pinned Git blobs. GM1 here uses the NMB common 939 MeV bare mass, so it must not be silently combined with the vacuum-Q static weak model. Thermodynamic identities and cold calibration do not establish an external finite-T forward comparison, a phase-complete EOS or all-domain stability. A spinodal example is preserved explicitly. Its Sommerfeld free-energy expansion is a narrowly tested asymptotic reference, not an all-temperature fit.

Author notes/checkpoint files retain their original sequence labels, including 'awaiting independent review'; the later review files and this index establish the accepted restricted scope. Numerical evidence does not measure omitted nuclear-physics uncertainty. No existing production implementation is replaced.

## Reviewed advanced reference slices

Subsequent commits add prescribed-density expansion/compression with explicit mechanical work; nonrelativistic recoil/Pauli density response with chemical KMS and sum rules; and the two-charge nuclear-only GM1 Hessian. Run the corresponding scripts in prescribed_density/, response_reference/ and rmf_phase/. Optional phase figures use an existing Matplotlib environment; the core calculations require only the already used NumPy/SciPy environment. The phase map samples nodes and is neither a binodal nor a crust boundary.

allowed_opacity/ adds leading allowed-current capture/inverse absorption only, with a derived absolute normalization. The heavy-mass limit matches the static reference without calibration. Crossed neutron decay is absent from this slice, so it must not replace a complete weak or cooling source. Phase-space recoil plus a static current is a declared leading-current ansatz, not a full relativistic or order-1/M calculation.

rmf_ad/ adds exact-primal first derivatives in density, temperature and proton fraction through a CPU oracle and an analytic implicit Jacobian. Use an existing JAX-capable environment; the recorded run uses JAX0.10.1. First-order forward/reverse AD and JIT are qualified on selected smooth warm states. Coupling derivatives, nested AD, phase derivatives and differentiation through an implicit stiff solver remain unsupported by this slice. Its callback intentionally rejects higher-order AD rather than manufacturing zero derivatives.

Each new slice carries its own author report and separate review. New archive manifests supplement the original source manifest; initial author notes can retain pending-review wording, superseded by later review files. No production promotion or merge follows from these research checkpoints.

## Matched FD matter and the crossed optical channel

crossed_opacity/ adds neutron decay and its three-body inverse using the common-mass FD response, with negative sum-energy/sum-momentum emission kinematics.39 author checks and independent physics/numerical reviews qualify the named leading-current channel. nr_baryon_eos/ supplies the matching free-FD common-mass EOS (nucleon offsets Q,0), with96 author and46 independent numerical checks plus a separate thermodynamic physics audit. These are matched free-quasiparticle building blocks, not a relativistic GM1 weak response.

capture_endpoint/ replaces raw electron-energy quadrature with an exact Ee=me+x² map, at selected48/48 controls.34 author and9 independent momentum-coordinate checks pass. This resolves the selected antineutrino1.8MeV integration error while preserving the old source and original failures. No all-domain guarantee follows: a later broader slab refinement found an unrepresentable transformed panel width. That case and a separately reviewed guard belong to the subsequent slab slice; this original mapped leaf does not handle that broader case.

The old crossed_decay.combined_antineutrino wrapper still calls frozen allowed_capture. Integration must explicitly select the mapped capture evaluator and crossed channel, rather than assume the repair propagates into that helper. Capture and crossed j/chi must use the same EOS factories and chemical conventions. No existing classical cooling element is silently upgraded by these new modules.

## Reviewed matched spectral slab and raw-coupling derivatives

quantum_slab/ explicitly composes guarded mapped capture plus crossed decay with the matched FD EOS for twelve selected spectral energies and two prescribed irradiated faces.514 author checks pass; independent physical collision-source and numerical ray/flux checks pass. A broad fallback initially bypassed invalid controls; preserved source/reports demonstrate the counterexample. The parent narrowed the fallback, restored validation at both entry points, reran506+8 valid checks and9 regression checks;20 independent post-repair checks pass with unchanged held-out physical outputs. The original CHECKPOINT.json is historical; CONTROL_REPAIR.md and REPAIRED_SOURCE_HASHES.json describe current source. Sources are column exchanges per area/per energy, not volumetric sources or time-evolved matter. No integrated luminosity or closed-fluid claim follows.

rmf_parameter_ad/ adds first-order derivatives in five raw nuclear couplings at fixedM939, using exact converged primals and analytic implicit rules.504 author checks plus7 independent physics and84 numerical checks pass, including new shifted couplings, an independent full species-Hessian gate and re-solved finite differences. Parameter-extended Gibbs-Duhem includes minus the canonical parameter-work derivative; omitting it doubles the vector-coupling pressure derivative. Raw-coupling changes are not recalibrated EOS uncertainties; external finiteT forward agreement, global phase selection and higher-order AD remain unqualified.

## Integrated charged-current moments and physical threshold study

quantum_energy_moments/ integrates the named complete electron-flavor charged-current inventory at two warm matched FD baths. The dilute602-check grid passes;14 initial moderate coarse-grid failures are retained, with70 targeted refined/remainder checks supporting the final summary.48 independent numerical and18 physics checks accept the qualified full-energy moments/net slab exchanges. Positive slab components are linear-source/incident-attenuation terms, not gross microscopic reaction counts. An initial-state matched-EOS diagnostic can yield rising temperature while total matter energy escapes, due to the composition derivative.

The explicit --resume path originally lacked state/source fingerprints. The author preserved that version and added full solved-state, source, environment, controls, energy-edge and result validation;31 author and8 independent cache tests pass. Legacy partials are intentionally rejected rather than silently migrated. The heavy-static integrated-inventory gate remains excluded after an underflow failure; selected-energy static checks do not replace it.

urca_threshold/ supplies nine independently reviewed spectral capture/inverse points under a declared constant-offset NR quasiparticle family.48 author gates,11 independent physics and80 numerical review checks pass. Offsets are set once per composition atT0 and held fixed in each temperature scan; different compositions are different local Hamiltonians. The independently verified finite-energy cold absorption onset distinguishes true thermal opening from an incident neutrino simply exceeding the cold onset. The failedT=.01MeV extension is preserved and excluded. No universal width, total Urca rate, interacting GM1 threshold or cross-family composition derivative is implied.

### Compressed raw evidence

Large new JSON reports are stored losslessly as .json.gz, with original/stored SHA256 and sizes in ARCHIVE7_EVIDENCE_MANIFEST.json. Source and small reports remain readable. From this research directory run:

    python restore_archive7_evidence.py --verify-only
    python restore_archive7_evidence.py

The first checks all compressed bytes and original hashes; the second restores exact JSON filenames needed by report/figure readers. Neither overwrites a changed existing file. Original failed grids and pre-repair snapshots retain their original status. Fresh physics drivers regenerate their own measurements; decompression is only for archived-evidence inspection.
