# Audited execution dependencies and replacement boundaries

Static source audit only: none of these modules was imported here. `SOURCE_INVENTORY.json` supplies literal imports, public signatures, call-bearing import-time statements and source hashes for29 modules. This diagram resolves the physical paths behind selected dynamic imports; it is not an executed exhaustive dependency proof. Package-version/minimum-runtime/system-library dependencies remain separate.

```
EC moments electron_capture.py
  -> relativistic_capture_oracle/events.py (independent gamma/current/invariants)
  -> relativistic_free_eos/free_baryons.py -> thermodynamics/lepton_reference.py
PC moments author moments.py
  -> positron_capture/pc_kernel.py -> current/vacuum/tensor + free EOS
PC independent reference review_pc_moments/reference.py
  -> NumPy/SciPy; own vectorized scalar polynomial/constants
beta moments hot_moments.py
  -> generic beta = relativistic_crossed_contract/beta.py -> full current/vacuum
  -> generic kernel = relativistic_crossed_thermal/kernel.py
       -> generic tensor = relativistic_tensor/tensor.py -> current/vacuum
       -> generic free_baryons -> lepton_reference
hot EC spectrum -> EC moment module + tensor + full current/vacuum + leptons
hot PC spectrum -> same geometry modules + independent PC scalar amplitude
hot beta spectrum -> hot beta moment/conditional + crossed kernel/current
spectral total consumers -> their spectrum modules + pinned actual JSON node chunks
transparent source coupling -> named EC/PC/beta, independent matched-state EOS
optional capture fit -> collect.py -> detailed actual source/EOS preparation
```

The new package must remove ROOT-relative paths, sys.path insertions, generic `beta`, `kernel`, `current`, `vacuum_cc`, `lepton_reference` module aliases, process `dont_write_bytecode` changes, and import-time filesystem hash/report reads. Restoring selected aliases does not make the current campaign graph a portable pure package. Source file verification remains useful test/provenance admission, not an installed runtime filesystem contract. No loader should import a driver with top-level calculations or write reports.

Do not merge independent scalar and literal-spinor implementations into one test that merely calls the same helper twice. Keep an independently derived test oracle (including the reviewer PC polynomial) with explicit convention comments/attribution; package runtime must not depend on reviewer directories. The author PC event kernel and scalar reference have different quadrature/maps and are not interchangeable evidence without matching all state/current controls.

The first portable phase is self-contained for chemical baths, using NumPy/SciPy and immutable physical constants/current records. Fresh net-charge/particle density factories are a later optional bridge to merged reviewed EOS components, explicitly supplying physical campaign masses to avoid the slightly different NMB default constants. One current provider receives on-shell species masses and a separately explicit weak normalization mass; no global monkeypatch or phase-specific constant injection.

Current full scalar polynomials duplicate form factors across existing modules; a portable reference should centralize the source-labelled mathematical model while keeping independent literal-current tests and immutable source benchmark rows. Changing to centralized runtime code is a forward-regression gate, not assumed bitwise identity. Fits, EOS/rates and spectral consumers must use the same model fingerprint, and no function may silently substitute stationary allowed rates when expensive detailed quadrature stops.
