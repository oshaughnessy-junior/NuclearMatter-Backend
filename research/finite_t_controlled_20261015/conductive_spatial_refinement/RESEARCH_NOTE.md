# Selected nonlinear thermal-slab spatial refinement

**The previously open dynamic spatial gate now has a selected-case self-convergence study with independent time-error separation:54 main and8 supplementary checks pass.** Three nested spatial grids use compatible cell-averaged conserved energy, actual temperature-dependent EOS/conductivity, and the frozen conservative solver. Independent review remains required. This is not a universal continuum proof or qualification of strongly nonlinear/other transport regimes.

No frozen slab, transport, EOS or review source was modified. Source hashes are recorded before and after the experiment. There is no conductivity interpolation, emulator or fixed-coefficient replacement.

## Case and conservative initialization

The same fixed-background electron model uses T0=.041MeV, ne=5.3e−8fm^-3,Z=2 and the original open-circuit screened-Born conductivity. External ions are static and supply no heat capacity. The isolated slab has zero boundary flux and initial smooth temperature

\[
\theta(\xi)=1+.04\cos(\pi\xi),\quad0\le\xi\le1,
\qquad\tau_{end}=.0015.
\]

The evolved variable is the actual normalized EOS energy

\[
W(\theta)=[u_e(T_0\theta)-u_e(T_0)]/(C_{V0}T_0),
\]

not temperature or a pointwise sample of energy. Each of16 finest initial cells uses Gauss quadrature of W(theta(x));8 versus12 quadrature nodes differ by at most2.56e−13. Coarse8/4-cell initial energies are **exact arithmetic volume projections** of those16-cell averages. Independently integrating the coarse cells agrees within1.76e−13. Initial EOS temperature inversion residuals are below1.48e−12.

Thus the nested grids represent the same averaged energy initial condition. Coarse temperatures are obtained by energy inversion, not by averaging fine temperatures or sampling the center. Cell FD entropy is evaluated after that inversion; no equality of initial entropy across grids is asserted.

The frozen explicit-Euler finite-volume solver evolves

\[
\partial_\tau W=\partial_\xi[k(\theta)\partial_\xi\theta],
\quad k=\kappa(T)/\kappa_0,
\]

with harmonic face conductivities and a guarded CFL based on max(k)/min(Cv/Cv0). Every changed state uses actual EOS and transport quadratures. The existing exact-state cache validates source/environment/controls and uses exact floating-point temperatures; no fit or interpolation supplies feedback.

## Spatial comparison: conserved cell averages

All three grids4/8/16 initially use16 time steps, Delta_tau=9.375e−5. At the same final time, fine **W values** are conservatively projected onto the coarse cells, then differenced. Norms are mean absolute, root-mean-square and maximum cell energy differences; they are self-convergence differences, not absolute errors against a known exact solution.

| Comparison | L1 | L2 | Linf |
|---|---:|---:|---:|
|4 versus projected8|1.4031406e−5|1.5304564e−5|2.1808115e−5|
|8 versus projected16|3.5769283e−6|3.9861109e−6|6.2169893e−6|
|Measured order log2(Dcoarse/Dfine)|1.97187|1.94091|1.81058|

The measured rates are consistent with the selected coarse-grid approach toward a second-order spatial discretization; the Linf rate is explicitly lower than2 and is not rounded away. A three-grid measurement is not a guarantee of asymptotic order throughout an arbitrary state rectangle.

## Independent temporal separation

The16-cell grid is independently evolved with16,32,64steps. Its step-halving differences are:

| Comparison | L1 | L2 | Linf |
|---|---:|---:|---:|
|16 versus32steps|8.6294467e−8|9.5844170e−8|1.5332439e−7|
|32 versus64steps|4.3125862e−8|4.7898697e−8|7.6613943e−8|

The ratios are approximately.5, matching first-order time stepping. The first-order estimate2*D16vs32, projected to each spatial-comparison grid, is at most **4.81% of the8vs16 spatial differences** and1.23% of4vs8. This is a measured Richardson estimate, not a rigorous error bound.

A separate experiment halves the time step on the4- and8-cell grids as well, so both spatial differences can be recomputed at32steps using the existing16-cell32step solution. Their changes are at most **.0612%** of the original spatial residuals across all norms. This directly checks that the observed spatial differences are insensitive to the chosen time step; it is separate from merely citing a stability CFL.

## Nonlinearity and physical controls

Actual source evaluations at theta=.96,1,1.04 give

| theta | kappa/kappa0 | Cv/Cv0 | Diffusivity ratio k/(Cv/Cv0) |
|---:|---:|---:|---:|
|.96|.95977477|.96015060|.99960857|
|1|1|1|1|
|1.04|1.04025390|1.03983026|1.00040741|

The EOS and conductivity each vary across the perturbation. Their ratio changes only about.08% across the full range, so this is deliberately a **mild nonlinear case**, not a stress test of large transport changes. The energy initialization/inversion and actual variable coefficients remain necessary and are not replaced by a linear approximation.

All runs preserve the temperature bounds, have nonnegative actual entropy increase and close isolated material energy reconstructed from the EOS within4.31e−13 normalized units. For16cells64steps the entropy increase is1.1601458580942859e−5. Boundary energy and entropy fluxes are exactly zero in this selected isolated case; the separate reservoir-ledger qualification remains in the frozen parent slab leaf.

Reference physical scales are unchanged: choosing L=100µm gives t0=1.1580616e−7s, so this short duration is1.73709e−10s. The Fermi-surface mean free path is8.8066e−10m, Kn=lambda/L=8.8066e−6; the finest cell width is6.25µm, much larger than that representative mean free path. These are selected-state diagnostics, not every-energy kinetic error bounds.

Local FD energy redistribution is still **imposed rather than derived**. The Lorentz operator alone cannot thermalize the isotropic energy distribution. The missing physical thermalization kernel and its effect on transport remain open. This study adds neither electron-electron collisions, ion heat capacity, electromagnetic/hydro evolution nor realistic crust/core physics.

## Runtime, provenance and preserved failure

The main numerical study took58.11s, with individual evolution experiments1.80,3.48,6.96,13.29,25.84s. The independent coarse-grid time-halving experiment took10.73s. Each is under120s; no runtime-budget stop occurred. The main study used4348 actual EOS evaluations and1953 actual transport evaluations.

`initial_averaging_failure.json` and its exact source preserve a Python generator-parenthesis exception before any evolution. The averaging expression was corrected without changing quadrature, physics or tolerances. No numerical gate failed after that correction.

`time_separation_results.json` pins the original result-data hash and verifies all frozen physical source hashes before using saved comparison arrays. It computes new changed-state coarse-grid trajectories; saved physical rates are not substituted for feedback. Root/source dependencies remain unchanged.

The selected dynamic spatial self-convergence gate is ready for independent review. Broader nonlinear amplitudes/states, alternate physical conductivity, more spatial grids or an independent continuum solver remain later qualifications. No NMB production or existing conductivity model is replaced.

Reproduce from repository root, in order:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/conductive_spatial_refinement/refine.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/conductive_spatial_refinement/time_separation.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/conductive_spatial_refinement/coefficient_range.py
```
