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
| `core_conduct` | `con_core_lep`, lepton-only `con_core` sum (ICON_CORE = 2) |
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
| 3 | `nu_n3p2_B_pbf` | gap v = sqrt(1-tau)(0.7893 + 1.764/tau) | 1.188/tau, as NSCool's Urca routines (src) |
| 4 | Urca vs pair breaking | zone isf counts as 1S0 in Urca and brem, 3P2 in pair breaking and C | one gap per zone |
| 5 | `specheat` | proton C ramps from the reduced value at Tc to normal at 1.1 Tc | BCS jump at Tc |
| 6 | `con_e_phon_ion_GYP` | nuclear composition from the Oyamatsu smooth model, not the crust table | (src) |
| 7 | `con_crust` | impurity scattering in the solid only | (src) |
| 8 | `NSCool_READ` | ETA divided by the fixed factor 0.443 before PCY97 | PCY97 Eq. A5 defines a gravity-dependent eta; see normalization below |

## Core transport and boundary coverage

`core_conduct.lambda_core(T, kfe, kfm, kfp, mstp, tcp)` implements the
NSCool 2009 electron–muon thermal conductivity, including the corrected
proton-pairing scattering fit. It uses local temperature and Fermi
momenta, with proton Landau/bare mass ratio. It returns scalar, unmagnetized
conductivity in erg cm^-1 s^-1 K^-1; inputs broadcast under JAX.

`zone_terms` now returns conductivity in core and crust zones. With
`rho_b = rhocore`, the last crust-march step uses actual core conductivity
and the star's proton gap at `icore`, replacing the former crust proxy.
`crust_march` therefore requires the star's `tc` keyword. Larger `rho_b`
remains outside this crust-blanket approximation and is refused.

This reproduces a named reference prescription, not universal core
transport. NSCool calls its baryon conductivity routine but deliberately
returns only the electron and muon sum; neutron conduction can be important.
The proton-superconductivity fit uses the older Pippard-limit screening.
[Shternin & Yakovlev 2007](https://arxiv.org/abs/0705.1963) describes the
lepton model; [Shternin 2018](https://arxiv.org/abs/1805.06000) improves
screening outside that limit. Add such an alternative separately from
reference-parity work. Hyperons and magnetic anisotropy are excluded.

The exactly absent-muon branch is preserved. Derivatives at species
onsets and the pairing transition are outside the smooth-AD contract;
scalar values at Tc match the reference. Gap-helper clipping keeps
unused branches finite.

Differentiate in log eta: at eta = 0 the gradient is 0 while the one-sided
derivative diverges. `ZoneDump.eta_nscool` is PCY97's eta (ETA / 0.443).

## Envelope normalization and provenance

The inputs use two conventions:

- `NSCool.run(eta=x)` writes the raw NSCool `ETA` and the reader converts it
  to `eta_pcy = x / 0.443`.
- `teff_local`, `isothermal_sums`, and `isothermal_curve` take `eta_pcy`
  directly. Passing the same number to both interfaces does not describe the
  same envelope. `ZoneDump.eta_nscool` is already converted; do not divide it
  again.

[Potekhin, Chabrier & Yakovlev (1997), Appendix A, Eq. A5](https://arxiv.org/pdf/astro-ph/9706148)
defines `eta_pcy = g_s14**2 * DeltaM / M`. The same convention appears in
[Page et al. (2004), Fig. 14 caption](https://arxiv.org/pdf/astro-ph/0403657).
Here `g_s14` is local surface gravity in units of 1e14 cm/s^2 and `DeltaM`
is the light-element/accreted-envelope mass. These papers establish the eta
convention; the caption does not specify the numeric constant 0.443.

Andrew Steiner explained the normalization on October 2, 2026: NSCool's
input parameter represents the envelope mass fraction, with 0.443 interpreted
as `1 / g_s14**2`. The source inspected at NSCool commit
`ed41c1e791897ff47c6bff34822c8c24026309a6`, `src/NSCool.f:309`, confirms the
fixed `eta=eta/0.443d0` operation. This implies a reference gravity
`g_s14 = 1 / sqrt(0.443)`, approximately 1.5025; the inspected line does not
recompute that factor from the current star's mass and radius. The exact
reference stellar model behind this constant remains unverified.

For an intended physical mass fraction `f = DeltaM/M` at a chosen mass and
radius, compute `eta_pcy = gs14(mass_msun, radius_cm)**2 * f`. To reproduce
that envelope through the raw NSCool interface, supply `ETA = 0.443 * eta_pcy`.
This is an input-convention conversion, not a change to NSCool physics. A raw
`ETA=f` is a physical mass fraction under the reference-gravity interpretation,
not generally at arbitrary mass and radius. Keep this distinction when varying
stellar models or differentiating their cooling curves.

The port retains the NSCool conversion for forward-reference comparisons.
Its envelope formula is nonmagnetic; changing eta normalization does not
validate other envelope prescriptions or extend the fixed-background cooling
model. Runtime NSCool tables/build dependencies and supported options remain
as listed above. Steiner supplied access and clarification; downstream papers
should cite the original physics and code provenance and discuss scientific
contributions with the collaborators. Documentation does not settle authorship.
