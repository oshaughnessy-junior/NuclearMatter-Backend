# Tabulated nuclear-EOS rotating equilibria

`nmbackend.rotating.tabulated` couples an existing positive, monotone `EOSTable`
to the pinned external RNS solver without imposing a polytropic closure. This
provides realistic nuclear-EOS stationary rotating backgrounds and baryon-density
maps for a rotating science case, such as conserved-baryon spin-down and access
to direct-Urca thresholds. It does not add a differentiable equilibrium primitive,
a full-GR mode solver, a thermal evolution solver, or a rotating-stability test.

## Build and solve

RNS remains external: use cgca/rns commit
`87ea68246ee990256f94aa91a0506f54d78dd8d0`. Its `main.c` is checked against the
qualified SHA256 before generating a separate export driver. The source tree is
not modified. Source, driver, and executable hashes are saved in `build.json`.
No dependency has been added.

```python
from nmbackend.rotating.tabulated import (
    C_CGS, MEVFM3_TO_ERGCM3, build_tabulated_rns, solve_tabulated,
)

# table is an actual nmbackend.eos.base.EOSTable, built by the chosen EOS model.
# Fresh build/run paths are required; existing results are never overwritten.
executable = build_tabulated_rns(rns_source, build_path,
                                radial_points=129, angular_points=65)
i = 300  # select an interior EOS row appropriate to the science question
central_energy = float(table.eps_mevfm3[i]) * MEVFM3_TO_ERGCM3 / C_CGS**2
spherical, rotating = solve_tabulated(
    executable, run_path, table,
    energy_center_g_cm3=central_energy, axis_ratio=0.9,
)
print(rotating.metadata['Mass_msun'], rotating.metadata['frequency_hz'])
print(rotating.metadata['baryon_number'])
```

The control is a spherical solve at the same central total energy density; the
second model continues in polar/equatorial coordinate axis ratio, supported here
for 0.6 through 1.0. A fixed central density pair does **not** conserve baryon
number. A conserved-number science sequence must separately solve for central
density at each axis ratio and verify the resulting `baryon_number` residual.

The native table has four columns: total energy/c² in g/cm³, pressure in
dyne/cm², c² times dimensionless pseudo-enthalpy in erg/g, and baryon number
density in cm⁻³. All input rows are retained, up to 8192 rows; the generated
driver expands the original 200-row allocation. Nonfinite, nonpositive, and
nonmonotone tables are rejected. Density jumps/pressure plateaus therefore need
a separate treatment rather than being silently interpolated by this adapter.

## Geometry and units

Geometry arrays retain native scaled RNS coordinates. Let L0 be metadata
`length_scale_cm`. The spatial metric is

    dl² = exp(2α)(dr²+r²dθ²) + exp(γ−ρ) r²sin²θ dφ².

Here r has the native length unit L0. Physical radii in cm are r×L0. `rho` is
a metric potential, not a mass density. `gama` is γ and `alpha` is α. Array
`omega` has units c/L0. Metadata `Omega` is normalized to that same unit so the
independent velocity check is

    v²/c² = [(Omega−omega) r sqrt(1−mu²) exp(−rho)]².

This normalization is essential: native RNS exports its global Ω in rad/s while
its frame-dragging array remains scaled. The separate keys `Omega_rad_s` and
`frequency_hz` preserve physical spin. `Mass_msun`, `Mass_b_msun`, `R_e_km`,
and `J_g_cm2_s` expose physical global quantities. The original native header is
retained under `native_header`; raw global `Mass` and `Mass_b` remain in grams.
The meaningless native default `Gamma` is removed for tabulated backgrounds.
Do not pass these backgrounds to the existing polytropic mode closure.

`energy` and `pressure` are scaled by 10¹⁵ g/cm³ times c²; `enthalpy` is
dimensionless; `velocity_sq` is the ZAMO fluid speed squared divided by c².
`fields['n_fm3']` uses native four-point polynomial interpolation of log n versus
log total energy density and vanishes outside matter. It supplies a density
bridge to independently computed EOS composition; RNS itself does not solve
weak equilibrium or composition.

The external source uses legacy c=2.9979×10¹⁰ cm/s, G=6.6732×10⁻⁸ CGS, and
m_b=1.66×10⁻²⁴ g. These are explicit provenance, not backend CODATA values.
Solar-mass conversion uses the backend IAU nominal mass. Native baryonic mass
uses a different per-baryon normalization from backend 939 MeV; use
`baryon_number=Mass_b_native_grams/1.66e-24` when that distinction matters.
Conserving native baryonic mass within one build is equivalent to conserving
number.

## Surface and executed forward gates

The driver uses the EOSTable's lowest enthalpy, pressure, and energy as the finite
surface boundary, matching the backend TOV table truncation. It does not continue
below the supplied table or add a physical crust or atmosphere. This common
boundary is required for a controlled comparison. Publication-quality results
still need crust, grid, EOS interpolation, and model-systematic checks.

A real 500-row GM1 RMFT table was exercised on cardassia on 2026-11-09 with the
129×65 RNS grid. At central baryon density nearest 0.5 fm⁻³, the spherical RNS
control and NMB `solve_structure` at the same exported central enthalpy agreed
with relative mass error +2.4075884×10⁻⁵ and radius error −1.3114131×10⁻⁴.
Native four-point log interpolation differs from the backend thermodynamic
Hermite EOS representation, and constants differ as described above.
The actual q=0.9 model returned M=2.1376320 solar masses, R_eq=14.0772119 km,
and Ω=3969.3916 rad/s. This is a forward gate and rotating smoke configuration,
not a conserved-number science sequence or a precision convergence claim.

An independent Python integration of the density/geometry product checked

    N_b = 4π L0³ ∫ ds dmu n_cm⁻³ (1−v²)^−1/2
          × exp[2α+(γ−ρ)/2] r_e³ s²/(1−s)^4.

Simpson quadrature on the original uniform compact s grid reproduced native
`Mass_b/m_b` with residual +2.858×10⁻¹¹ for both spherical and q=0.9 models
(the native rounded π explains this floor). The exporter omits the final
vacuum node; restoring its zero integrand at s=0.9999 preserves Simpson parity.
Trapezoid residuals were −4.00×10⁻⁵ and −6.91×10⁻⁴. Agreement of this integral
validates the density/volume convention; it is not an independent equilibrium
solver comparison. Threshold-masked accessible fractions require their own
resolution checks, especially near onset.

The rotating science deliverable can therefore build conserved-number sequences
and compare density maps against composition-derived Urca thresholds. That
future deliverable must state the composition model, conserved-number tolerance,
and threshold-mask refinement, and must distinguish a cooling-accessibility
proxy from an actual cooling curve. Rotating modes additionally need a real-EOS
perturbation closure and independent mode benchmarks.
