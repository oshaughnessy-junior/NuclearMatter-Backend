# Physical inelastic νe–e± plane-parallel slab: finite S₂ reference

**Author freeze, independent review pending.** This adds actual inelastic neutrino–electron/positron scattering to a small plane-parallel streaming configuration, with fixed FD pair bath and explicitly prescribed incoming radiation. It is a finite-energy, finite-angle, finite-volume transport ansatz, not hydro, a grey relaxation surrogate, a continuous angular solution or an energy-continuum result. The selected physical source and time checks pass; three failed coarse azimuth gates remain preserved.

## Bath, current and geometric ansatz

The prescribed bath is uniform T=2MeV with net electron density3e-8fm^-3 and µe=.17446333200826397MeV. Actual equilibrium pair densities are approximately ne−=2.034286367843283e-7 and ne+=1.734286367843283e-7fm^-3; independent adaptive momentum integrations check both and the net charge. No material energy/composition is advanced. Muon targets, photons, nucleon scattering, plasma correlations, QED/EW precision corrections, flavor conversion and oscillations are absent.

The elementary function is the previously reviewed `neutrino_lepton_scattering/scatter.py` massive free Fermi-current kernel D(E,F,c)=dΓ/(dF dc), per incident one-active-helicity νe, before final-neutrino blocking. Its electron flavor includes CC–NC interference, GF but no Vud, physical me and actual e± FD target/hole factors. Both directions and both targets are computed as detailed phase integrals. Current/absolute normalization qualifications remain in the elementary scattering/review leaves; this slab does not derive a new weak current. [Guo et al. primary leptonic kernel, Eqs1–4 and AppA](https://arxiv.org/html/2006.12051v2).

Radiation is axisymmetric around the spatial z axis. Use four Gauss energy nodes on **0–24MeV** and S₂ directions µ=±1/sqrt3 with weights v=1 each. There are three uniform upwind cells in the selected slab. The light-crossing scale is L/c=.05s, hence L=1.49896229e9cm using c=2.99792458e10cm/s. This deliberately controlled length is not a neutron-star atmosphere/geometry inference. Evolve for.06s. Initial radiation is isotropic FD with Tr=2MeV, µν=−.7MeV. Only incoming directions are prescribed: at z=0, µ>0 has FD Tr=3MeV; at z=L, µ<0 has FD Tr=1.3MeV, both with µν=−.7MeV. Outgoing values come from the adjacent evolving cell. No effective radiation temperature is imposed after initialization.

## Angular phase measure and literal physical transitions

For outgoing polar direction µβ and relative azimuth φ,

```
cαβ(φ)=µαµβ+sqrt[(1−µα²)(1−µβ²)] cosφ,
Dbar_iα,jβ=(1/2π)∫dφ [D_e−(Ei,Ej,cαβ)+D_e+(Ei,Ej,cαβ)].
```

D already integrates the outgoing azimuth for an incident-axis representation. The general rate per solid angle is D/(2π); the axisymmetric final distribution therefore gives outgoing measure **wj vβ Dbar**, without an additional1/2 and without another Ej² DOS. The initial one-helicity density measure is

```
Wiα = [Ei²/(2π²(hbar c)³)] wi vα/2,
Aiα,jβ = Wiα wj vβ Dbar_iα,jβ.
```

The factor vα/2 is the initial normalized angular average; summing it recovers the usual isotropic density. Every off-mode direction is independently evaluated, including equal-energy scattering between **different** µ directions. The collision uses updated forward fiα(1−fjβ) and reverse fjβ(1−fiα) at every ODE stage. Exchange of energy and polar direction is retained; it is not replaced by an isotropic scalar opacity.

Exact full-mode self-transitions `(Ei,µα)=(Ej,µβ)` are omitted **before** azimuth integration because an axisymmetric occupation has exactly identical occupation/hole products there and its collision contribution is identically zero. This avoids assigning an artificial finite activity to the coincident E/µ azimuth line, whose pointwise forward small-angle kernel can be singular. It does not assert D(E,E,c)=0 or remove equal-energy polar redistribution. No total scattering-frequency claim is made from this off-mode discrete generator.

Measured physical reciprocity gives `Aab/Aba=exp[(Ea−Eb)/T]`; it is not enforced by generating inverse entries from forward ones or symmetrizing K. Same-T FD occupancy at any chemical potential is therefore stationary to measured integration error. Paired signed flux uses stable expm1 subtraction, conserves number and separately accumulates material energy as minus the radiation energy change. Collision entropy `sum J log(F/B)` is nonnegative and agrees with radiation entropy derivative plus thermostat heat/T at anisotropic test states. **No complete open-system entropy balance including irradiation reservoirs or numerical streaming entropy is claimed.**

## Streaming, boundary and thermostat ledgers

In each cell use the actual upwind face flux µf and

```
df/dt = −(c/L) Nc [µf_face,right−µf_face,left] + C[f].
```

No limiter/clipping or source relaxation fit is used. The physical spectral kernel is fixed because the bath is prescribed, while both radiation holes change with f. Integrate the full nonlinear ODE using DOP853; a different explicit Heun time rule supplies the independent time check.

All reported extensive ledgers are **normalized per slab volume**. Incoming/outgoing number and energy face fluxes are divided by L, then integrated in time. The result is fm^-3 or MeVfm^-3, matching the volume-averaged radiation change. Thus

```
Delta Nrad = Nin−Nout,
Delta Erad + Qthermostat = Ein−Eout.
```

Thermostat heat is energy delivered to the fixed pair bath, which its external controller removes to keep T and µe prescribed. It is not stored EOS energy of an evolved material. The prescribed irradiation reservoirs have combined energy change `−(Ein−Eout)`; adding radiation, thermostat and boundary-reservoir ledgers closes the total. Each νe carries electron lepton number+1, so the number boundary ledger is also the radiation-plus-reservoir electron-lepton ledger. Scattering creates no charged-lepton composition or net lepton source in the bath.

## Forward checks, failures and selected numerical evidence

First: a zero-collision calculation matches the **exact semi-discrete upwind chain** with constant incoming boundary values. For a uniform initial field, the cell at upwind distance j has `fB+(f0−fB)e^(−at) sum_{k=0}^j(at)^k/k!`, with a=|µ|Nc/(L/c). This is an advection-discretization oracle, not an exact continuous slab solution. Equal bath-temperature isotropic FD radiation with matching boundaries is stationary, and fresh anisotropic occupations pass independent dense directed-event/paired number, energy and entropy comparisons.

Initial azimuth16→32 and32→64 rules FAIL the unchanged3e-6 pointwise edge gate: maximum changes2.20899% and2.00944e-4 respectively. The phi32→64 slab-heat comparison also FAILS at7.31083e-6 relative. These **three failures** remain in results.json with their complete grids, source and trajectories; passing KMS or a small energy residual does not reclassify them.

New actual phi128/inner32/tail80 versus phi256/inner40/tail90 grids pass the same3e-6 full off-mode edge gate: maximum5.6340e-7. Slab heat changes2.24075e-8 relative and occupations1.46e-11 absolute. The most difficult actual same-direction energy edges are checked against a separate analytic-target-azimuth/adaptive transverse-shell calculation with adaptive outgoing relative-azimuth integration; maximum discrepancy1.014e-8. Earlier three selected physical edges, including equal energy/opposite µ, are also retained. No weak prefactor is tuned to these comparisons.

Final selected physical values:

| Quantity over.06s | Value per slab volume |
|---|---:|
| Incoming number |9.27611094892e-8fm^-3|
| Outgoing number |5.10743118883e-8fm^-3|
| Radiation number gain |4.16867976009e-8fm^-3|
| Incoming energy |8.31562615410e-7MeVfm^-3|
| Outgoing energy |3.60900017852e-7MeVfm^-3|
| Radiation energy gain |4.41180296167e-7MeVfm^-3|
| Thermostat heat gain |2.94823013914e-8MeVfm^-3|

The net incoming boundary energy is4.70662597558e-7MeVfm^-3. Radiation and bath receive it with their stated signs. Absolute number and energy residuals are9.93e-23fm^-3 and−3.18e-22MeVfm^-3; conservative closure is algebraically necessary, not evidence of continuum accuracy. Occupations remain between3.52e-6 and.303635. The trajectory differs nontrivially from zero-collision streaming.

The independent explicit Heun64/128/256 time study measures orders2.0119 and2.0061 against DOP853; its final thermostat heat differs by3.29e-6 relative. Every explicit predictor calls the physical collision with strictly interior f; no clipping is used. Fine DOP853 time-control comparisons also pass. Changing **three to six spatial cells changes thermostat heat2.79875%**. This is a sensitivity result, **not** a spatial convergence pass. S₂ angular resolution and four energy nodes are similarly controlled finite ansatz choices, not proven continuum accuracy.

The earlier closed spectral box has hot heat sensitivity~19.5% for24→48MeV and1.16% for36→48MeV. Those numbers are not transferred as an error estimate for this different slab problem, but they demonstrate why this slab's24MeV ceiling cannot be called a converged energy continuum. Missing energy-range transitions are excluded from the collision generator. Spatial boundary irradiation is independently open and explicitly accounted for; it does not restore those omitted energy transitions.

Author gate accounting: **55 passing assertions across the initial and fine batches**, with **three coarse azimuth failures retained/excluded**. Fine accepted_results.json has23/23 passes; initial results.json has32passes/3fails. These are related algebra/control gates, not55 independent physical benchmarks. Initial physical batch5.03seconds and refinement18.31seconds; each grid/source/evolution experiment stays below120seconds. No heavy build, dependency install, upstream edit or production implementation is involved.

## Reproduction and remaining gates

Run fresh `python check_slab.py`, then `python refine.py` with the existing campaign NumPy/SciPy interpreter. Both write only this owned leaf. `slab.py` and `reference.py` are function modules loaded under unique names; no frozen study/validator is imported. Frozen elementary sources are fingerprinted at every edge; source/environment packaging is still trusted research execution, not the separately reviewed process-isolated API or an OS security boundary. CHECKPOINT.json pins source, upstream hashes and raw evidence. `make_figure.py` reads saved fine outputs only; PNG/PDF and figure_data.json show actual four energy nodes and upwind face fluxes. Connecting lines guide the eye. The figure was visually inspected.

Next gates: independent review of angular/DOS factors and self-mode cancellation; separately refined spatial, S_N angular and energy-range observables; boundary-driven entropy accounting; more irradiation/bath states; then optional prescribed material feedback or other flavors. Do not infer weak uncertainty from quadrature agreement, continuum spectra from four nodes, LTE from forced source cancellation, or a full GRMHD-neutrino backend from this controlled slab.
