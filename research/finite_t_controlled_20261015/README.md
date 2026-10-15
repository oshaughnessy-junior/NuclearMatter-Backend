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
