# Independent physics review: coupled element and spectral slab

15 October 2026. Reviewed `fluid_element/fixed_temperature_element.py`, its technical note, serialized results and history-refinement summary, together with `spectral_slab/run_slab.py` and its results. No author files changed. This review supplements `PHYSICS_REVIEW.md` and accepts only the declared static, classical, noninteracting nuclear ansatz. Usage/account/window matched campaign guards at review start (2% used).

## Disposition

**Accept the three-state thermostatted transparent fluid-element demonstration and the single-state fixed-bath two-vacuum-face spectral slab as restricted research benchmarks. No blocking physics inconsistency was found in their coupling.** Neither demonstrates a self-cooling thermal evolution, a globally thermalizing closed box, finite-recoil Urca, neutron-star-core rates, or radiation hydrodynamics. Their qualifications make this limitation clear.

The numerics review reports the slab's 40 checks passing, including an independent blackbody-limit check. This review inspects the corresponding angular and energy measures independently and reads the recorded 12 element ODE runs plus six matched tolerance-refinement histories; it does not repeat those expensive integrations. A separate small calculation below bounds nucleon phase-space occupancy throughout every physical composition, rather than only at the root.

## Spectral slab: factors and physical interpretation

The one-helicity isotropic density measure is `D E² dE`. With axial symmetry, its directional measure is `D E² dE dmu/2`. For a homogeneous bath and two identical vacuum faces, the outward occupation is `f_eq[1−exp(−tau/mu)]` on either face, with `f_eq=j/(j+chi)` and `tau=(j+chi)L/c`. Thus

```
F_number,both(E) = c D E² f_eq integral_0^1 mu [1−exp(−tau/mu)] dmu
F_energy,both(E) = E F_number,both(E).
```

This is exactly the implemented factor. One face has half this flux. The thin limit is `D E² j L`, which is the spontaneous volume emission integrated over depth, and the thick active-mode limit is `c D E² f_eq/2` for both faces. It is incorrect to insert another spin factor, another angular factor two, or `chi` in place of `j+chi` in the attenuation. Multiplication by `1e39` converts the fm^-3 density measure to cm^-3 before obtaining cm^-2 s^-1 flux with `c` in cm/s. The sine-squared energy mapping includes its Jacobian correctly.

The emitted flux ratio to optically thin spontaneous emission is a well-defined escape diagnostic for this fixed bath. It is not a geometry-independent gray opacity and not the local emission rate of an evolving material. The material thermostat/composition reservoir replaces the absorbed/emitted radiation exchange; no closed matter-plus-radiation conservation is implied.

The exact antineutrino support gap is retained. Inactive modes have `j=chi=0`; with vacuum boundaries their occupations stay zero even at arbitrarily large slab length. This is compatible with the previous review's disconnected-mode finding and is not a failure of the thick active-mode limit.

The accepted example is `T=3 MeV, nB=1e-5 fm^-3, Ye=0.15`. Generic-function caveat: `energy_grid()` is qualified for this example, not every signed chemical potential or deeply degenerate state. Its upper cutoff does not explicitly track a large negative `mu_e` positron Fermi energy, and its internal breaks do not resolve every possible narrow charged-current Fermi edge. Before reusing it across a broad state grid, construct channel-aware cutoffs/panels and require state-specific refinement. This is not a blocker for the reported positive-mu warm example.

## Fluid element: variable transform and ledgers

At fixed `T,nB`, charge neutrality gives `Ye=n_e,net(T,mu)/nB`. The positive susceptibility makes the implemented transform `dot(mu)=nB F/(dn/dmu)_T` equivalent to `dot(Ye)=F`. The stability diagnostic also transforms correctly: `−F_mu nB/n_mu = −F_Y`. A finite local susceptibility is required; positivity of susceptibility alone does not guarantee a generic ODE trial stays within `0<=Ye<=1`, so the existing explicit domain checks remain necessary. The accepted trajectories stay in domain and approach the stable kinetic root from both sides.

The matter energy subtracts one constant proton rest energy while retaining `(1−Ye)Q`, common `3T/2` nucleon kinetic energy, the pair-equilibrated lepton energy and photons. Its composition derivative is

```
u_Y = −Q + mu_e + T n_T/n_mu
    = −Q + mu_e − T mu_T|n.
```

The code uses this derivative correctly. Because both density and temperature are fixed, photons and classical nucleon kinetic energy do not change during these runs. The signed thermostat power per baryon is `h_ext=loss_nu+u_Y F`; consequently `Delta u=integral(h_ext−loss_nu)dt`. The external heat reservoir is an essential part of the model. Removing the thermostat without adding a temperature equation would violate the declared physics. No additional Q source should be added.

Neutrino and antineutrino number accounting has the correct signs: `Delta Ye+N_nue−N_antinue=0`. Nonzero escaped energy and approximately equal continuing gross number emission at the kinetic root are expected. The equilibrium source root is distinct from chemical equilibrium with zero neutrino chemical potential, as the report explicitly records.

The recorded largest scaled residuals are **2.43e-9 for energy** and **5.62e-10 for lepton number**. These are residuals under the stated gross-exchange/absolute scales, not a universal relative error of tiny net sources. The six serialized histories change by at most **9.59e-10** under the reported ODE tolerance refinement. These results establish coupled execution and selected numerical consistency within the same physical ansatz.

## Classical-domain audit beyond the root

For common kinetic mass `M=939 MeV`, define `lambda=sqrt(2pi hbarc²/(MT))`. Because `n_i<=nB`, `nB lambda³/2` bounds either nucleon's phase-space occupancy for **any** `0<=Ye<=1`, including both initial compositions and the full trajectory. Independently evaluating this bound from the recorded states gives:

| T [MeV] | nB [fm^-3] | Upper bound on either n_i lambda³/2 |
|---:|---:|---:|
| 0.5 | 1e-8 | 5.94766e-5 |
| 2 | 1e-7 | 7.43457e-5 |
| 5 | 1e-5 | 1.88081e-3 |

For the slab state, the actual larger species occupancy is **3.43984e-3**. These are suitably dilute for the intended classical-nucleon demonstration. They do not bound recoil corrections, many-body uncertainty, or nuclear recombination. The ansatz deliberately excludes nuclei even where a more complete composition model might favor them; therefore these are controlled free-nucleon experiments, not a demonstration of full thermodynamic composition equilibrium. Nuclear interaction/composition uncertainty must not be inferred from numerical convergence.

## Next accepted increment

Proceed to a self-cooling fixed-density element with the same total-energy convention and positive frozen-composition heat capacity, then to one prescribed density history with compression work. Require explicit independent energy-derivative and ledger tests before claiming thermal evolution. A closed box remains useful only with the documented disconnected-mode constraint or a qualified redistribution kernel. Broad slab state sweeps need the channel-aware energy-grid refinement described above. No full hydro solver or production throughput optimization is needed for these increments.
