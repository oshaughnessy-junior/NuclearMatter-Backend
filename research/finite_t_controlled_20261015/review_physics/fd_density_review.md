# Independent physics review: prescribed-density free-FD element

15 October 2026. **Accept the reversible weak-off density experiment and thermodynamic equations in the declared free-FD model.** Weak-on density evolution remains a design, not a demonstrated result. No author files were changed. Reviewed `density_element.py` SHA256 `d539c8e2a86cb31c10d459d7f1a5609e8c0c2e1fa737256cb7a67e7ea04309b2`, `weak_off.py`, note, raw55-check results, checkpoint and matched underlying EOS.

## Density derivative and first law

The comoving element has fixed baryon number, externally specified density, and charge neutrality `nn=(1−Y)n`, `np=ne,net=Yn`. Its material energy includes neutron Q and total e± energy; a common nucleon rest contribution is subtracted. For each pair-equilibrated or fixed-number ideal species, `∂epsilon_i/∂ni|T=mu_i−T mu_i,T ≡ Ai`. Hence differentiation of `u=epsilon/n` gives exactly

`u_n=[(1−Y)An+Y(Ap+Ae)−u]/n`.

The electron term uses **net** charge and its total chemical potential; pair energy is retained in epsilon and Cv. The density derivative has units MeV fm³, not an energy density. The common nucleon offset cancels between the weighted A terms and u; the physical Q remains where composition changes matter. No density-dependent offset is introduced, so there is no missing rearrangement work.

An independently derived free-energy identity is `P=n² f_n`, `u=f−T f_T`, giving `n² u_n=P−T P_T` and `s_n=−P_T/n²` at fixed composition. Therefore the weak-off temperature equation is

`dT/dln n = T P_T/(n u_T)`.

Compression heats the tested state and work increases u; expansion reverses both. These are pressure work and finite-T EOS effects, with no imposed 3T/2 heat capacity or classical composition-energy approximation.

For optional photons, `epsilon_gamma=aT^4`, `Pgamma=epsilon_gamma/3`, `Cvgamma=4epsilon_gamma/T`. They add `−epsilon_gamma/n²` to u_n, add `4epsilon_gamma/(T n)` to u_T, and add nothing to u_Y. The author cycle is photon-free. Our held-out witness below deliberately enables photons, checking their density and thermal terms separately.

## Weak-source bookkeeping, not a weak-on trajectory

At prescribed density the correct material first law is `du/dt=P dn/dt/n²−q`, with `dY/dt=F`. Resolving u(T,n,Y) yields the documented `u_T dT/dt=(P/n²−u_n)dn/dt−u_Y F−q`. The independent entropy identity is

`T ds/dt=−q−(mu_p+mu_e−mu_n)F`.

The chemical term contains chemical potentials, whereas the temperature equation contains energy-composition derivative u_Y; interchanging these would be wrong. The documented equations use them correctly. Number moments have the one-helicity `E²/[2pi²(hbar c)³]` DOS, energy moments one additional E; escaping total energy includes the energy associated with composition/rest-mass changes already carried by the EOS. There should be no second manually added Q heating term.

The proposed split between a weak update and a reversible density move is first order and needs genuine changed-state sources and refinement. No such trajectory was computed here or in the author evidence. The ledgers `u+Eescaped−W=u0`, `Y+Lescaped=Y0` have correct signs for the named transparent electron inventory. Spectral source accuracy, split error and conservation residual are separate tests.

## New held-out entropy/work witness

`fd_density_probe.py` uses **T0=2.4MeV, n0=1.3e−4fm^-3, Ye=.27, photons=True**, outside the author's selected initial state. It imposes log-compression0.003. A bracketed entropy root solves each endpoint/midpoint without the author's Newton derivative or temperature ODE. A separately assembled three-point Simpson integral of `P/n dln n` is compared to the full EOS endpoint energy difference:

| Quantity | Independent result |
| --- | ---: |
| Final temperature [MeV] | 2.40436674093144 |
| Pressure work [MeV/baryon] | 0.0117019281864999 |
| EOS energy increase [MeV/baryon] | 0.0117019281865094 |
| Relative work/energy discrepancy | 8.08e−13 |

**10/10 checks pass**, including entropy preservation, positive work/heating, photon density/thermal/composition terms and invariance of u_n under an added common37MeV nucleon rest offset. Raw EOS calls, runtime and hashes are saved in `fd_density_probe_results.json`. The pressure-work comparison retains the original2e−9 author work tolerance. Its agreement is a targeted physical consistency test, not a new broad accuracy domain.

The first review-probe invocation had an unmatched parenthesis and failed before any calculation; it was corrected in the review driver. No author implementation or scientific gate was changed.

## Accepted scope and limits

The author compression-return cycle, separate signed work legs, full entropy roots and temperature ODE are consistent with a reversible comoving element in the specified free-NR-baryon/relativistic-pair-lepton EOS. The archived120s incomplete attempt remains incomplete; the76s retry changes root/quadrature algorithms, not physics. The declared microsecond history is an external time label and does not prove microscopic equilibration on that timescale. This leaf supplies no momentum/gravity/hydro solution, weak-on density feedback, interacting EOS or relativistic nucleon-current qualification. Those gates remain open.
