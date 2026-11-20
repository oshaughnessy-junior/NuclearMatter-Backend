# Scoped inventory naming/unit repair

The independent107-check review accepted the scientific recipe and identified an inventory naming advisory: integers formerly called `delta_material_Ye_per_event` and `delta_material_Ymu_per_event` were stoichiometric **particle-count changes**, not composition-fraction increments. The original interpretation was imprecise. All eight reviewed files, exact source/inventory/raw and original manifest/checkpoint are preserved in `pre_inventory_repair/`, with hashes in `PRE_INVENTORY_REPAIR_PIN.json`. The reviewer's original107-check report is untouched.

`reaction_inventory.json` now uses explicit `delta_Ne_per_event`, `delta_Nmu_per_event`, `delta_Np_per_event` and `delta_Nn_per_event`. Their original integer values are unchanged. Radiation Le/Lmu entries are separately named signed lepton-number counts. Units and conversion are explicit:

```
Y_i=N_i/N_B=n_i/nB
one directed event in volume V: delta Y_i=nu_i/(nB V)
directed number rate R_a [fm^-3 s^-1]: dY_i/dt=sum_a nu_i,a R_a/nB [s^-1]
radiation flavor-number source=sum_a nu_rad,a R_a [fm^-3 s^-1].
```

This reference computes only energy emissivity Q; it does not supply those physical event-number rates R_a. They cannot be inferred by substituting Q into the number-rate formula. The unit regression uses explicitly synthetic prescribed number rates, not a claimed MU number calculation.

`inventory.py` makes the inventory reproducible without importing the physical solver. `check_inventory_repair.py` has45 passing controls: all original integer values, every directed electric/baryon/flavor ledger, a nonzero mixed-flavor number-to-fraction example, one-event finite-volume normalization, density scaling and independent finite-volume count accounting. Scientific source, geometry, validator and numerical results hashes remain exactly unchanged; no physical numerical replay was needed. The original reviewed model coefficient and all admissions/limitations are unchanged.

The scientific107-check review remains the original acceptance record. This naming repair is frozen pending a **separate recheck** by that reviewer; it is not silently added to or substituted for the original107.
