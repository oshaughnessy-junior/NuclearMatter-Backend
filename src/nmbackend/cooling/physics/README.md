# nmbackend.cooling.physics: NSCool's physics in JAX

Paper 7, step C2. Each function follows the NSCool routine named below and
is tested zone by zone against `NSCool.zone_dump` (`tests/test_cooling_physics_*.py`).

```python
import numpy as np
from nmbackend.cooling import NSCool
from nmbackend.cooling.physics import Star, isothermal_sums, isothermal_curve
from nmbackend.cooling.physics.reference import GAPS
d = NSCool().zone_dump([1e8], mass=1.4)
s = Star.from_dump(d); tc = s.tc(GAPS["default"])
r = isothermal_sums(1e8, s, tc, eta=1e-9, rho_b=1.6e14)   # eta: PCY97's eta
c = isothermal_curve(np.logspace(4, 10, 961), s, tc, eta=1e-9, rho_b=1.6e14)
```

| Module | NSCool routines |
|---|---|
| `gaps` | `get_Tc`, `tc.f` fits, `get_kf_scaled` (nmbackend patch) |
| `core_nu` | `numurca_nucl`, `nubrem_nucl`, `nudurca_h`, `nu_1s0_pbf`, `nu_n3p2_B_pbf` |
| `crust_nu` | `neebrem`, `npb_new`, `nplasma`, `nubrem_crust_nn` |
| `heat` | `specheat`, `cvt_deg`, `cvion` |
| `conduct` | `con_crust` (ICON_CRUST = 3) |
| `envelope` | `fteff_acc` |
| `star` | the C1 driver's sums and crust march |

Only the options of our runs are ported (I_Neutrino_1, I_Conduct_32,
IOPACITY = 0, no heating, no hyperons or quarks). Two NSCool data tables,
`SF_suppression.dat` and `Nalpha.dat`, are read from the NSCool build at
run time, since NSCool has no licence.

## Where NSCool departs from its cited papers

We kept each of these. Rows marked (src) are read from the source, and we
have not checked them against the paper's equation.

| # | Routine | NSCool | Paper |
|---|---|---|---|
| 1 | `numurca_nucl` | proton branch has no threshold step | Yakovlev+2001: zero for k_Fn > 3 k_Fp + k_Fe (src) |
| 2 | `numurca_nucl`, `nubrem_nucl` | n and p reductions combined as min(R_n, R_p) | Yakovlev+2001 give joint factors (src) |
X
| 4 | Urca vs pair breaking | zone isf counts as 1S0 in Urca and brem, 3P2 in pair breaking and C | one gap per zone |
| 5 | `specheat` | proton C ramps from the reduced value at Tc to normal at 1.1 Tc | BCS jump at Tc |
| 6 | `con_e_phon_ion_GYP` | nuclear composition from the Oyamatsu smooth model, not the crust table | (src) |
| 7 | `con_crust` | impurity scattering in the solid only | (src) |
| 8 | `NSCool_READ` | ETA divided by 0.443 before PCY97 | PCY97 Eq. A5 (nmb-papers OPEN-eta-0443.md) |

## Known difference

With `rho_b = rhocore`, the C1 driver's last march step reaches zone
`icore`, where NSCool uses its core conductivity (C4). We use the crust
formula there, and T_in differs by up to 0.15%. Inside the crust the
march matches to 1e-8.
