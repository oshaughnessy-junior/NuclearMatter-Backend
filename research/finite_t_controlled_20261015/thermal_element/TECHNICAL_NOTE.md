# Isochoric transparent cooling reference: no thermostat

15 October 2026. Research prototype, pending independent review. The calculation evolves matter temperature and composition with escaping electron neutrinos/antineutrinos at prescribed constant baryon density. It composes the already reviewed adjacent ideal-lepton and static weak references without changing them. No hydrodynamic variables or radiation field are solved.

## Matter energy and reaction convention

Matter comprises classical, noninteracting free neutrons/protons with a common 939 MeV kinetic mass; a neutron-proton rest-energy offset Q; ideal finite-mass electrons/positrons in pair equilibrium; and locally trapped photons with zero chemical potential. The common nucleon rest energy is subtracted as a constant. This explicit energy convention is

```
u(T,Ye) = (1-Ye) Q + 3 T/2 + [epsilon_e+-(T,Ye*nB)+a_gamma T^4]/nB,
a_gamma = pi^2/[15 (hbar*c)^3].
```

All temperatures and chemical potentials are MeV, densities fm^-3 and u MeV per baryon. The electron chemical potential includes rest energy. Charge neutrality fixes `n_p=Ye*nB`, `n_n=(1-Ye)*nB`, and net electron density `Ye*nB`.

The rates include electron capture, positron capture and neutron decay; all produced neutrinos escape. Final-state neutrino blocking and incident radiation are absent. Electron/positron blocking/occupations and exact electron mass and Q are retained by the static kernel. Nucleon recoil, thermal velocities and Pauli blocking, mean fields, nuclei, muons and pairing are omitted. The common kinetic mass and finite Q are exactly the declared weak-reference approximation, rather than physical finite-mass kinematics.

Photons do not escape in this model: they share matter temperature and supply heat capacity. The ansatz assumes that electromagnetic interactions maintain a local common temperature and electron/positron pair equilibrium. Opacity/thermalization times for that assumption have not been calculated here. Neutrino transparency is also imposed, rather than derived from a chosen parcel size.

## Evolution equations and conservation

Let `F=(R_pc+R_nd-R_ec)/nB` be the net composition source and `q=Qnu/nB` the positive escaped neutrino energy rate per baryon. The first law for this fixed-volume element is

```
dot(u) = -q,    dot(Ye)=F.
```

There is no thermostat, compression work, heating prescription or extra nuclear Q heat. The Q contribution appears once in u and also consistently in the energy of the emitted neutrino.

The EOS derivatives at independent `(T,Ye)` are

```
u_T = 3/2 + Cv_e,fixed_net_density/nB + 4 a_gamma T^3/nB,
u_Y = -Q + mu_e - T (dmu_e/dT)_net_density,
dot(T) = [-q-u_Y*F]/u_T.
```

The ideal-gas identity `d epsilon_e/dn_e|T = mu_e-T*dmu_e/dT|n_e` includes pair contributions and the implicit neutrality constraint. It follows from the thermodynamic potential; a finite-difference step ladder independently checks the actual returned energy function before any trajectory is evolved.

The numerical state uses `(T,mu_e)` to avoid an expensive density inversion at every ODE stage. Applying charge neutrality gives

```
dot(mu_e) = [nB*F - (dn_e/dT)_mu*dot(T)]/(dn_e/dmu)_T.
```

The ideal susceptibility is positive on the tested states. At each stage the code checks physical composition, positive heat capacity, and a classical-nucleon phase-space occupancy bound. The local transformed RHS is separately checked to recover both requested charge and energy derivatives.

Accumulated escaped number and energy obey the global ledgers

```
u(t)-u(0)+E_nu,escaped(t)/baryon = 0,
Ye(t)-Ye(0)+N_nue,escaped(t)/baryon-N_antinue,escaped(t)/baryon = 0.
```

Both ledgers are evaluated on independent EOS states along the output history. The energy scale excludes the constant common baryon rest mass, so that a large constant cannot hide a thermal-energy error. Matter energy decreases; temperature need not be assumed monotone for arbitrary composition because composition can also store or release energy. No chemical or kinetic equilibrium is enforced during evolution.

## Bounded runs and acceptance tests

Two examples begin at `(T[MeV],nB[fm^-3],Ye)=(2,1e-7,0.3)` and `(5,1e-5,0.5)`. Each stops at the prescribed floor `T=0.7*T_initial`, not at a fitted equilibrium. The broad evaluation guard is `0.20<T<10 MeV` and `max(n_n,n_p)*lambda_nucleon^3/2<0.02`. The script reports the actual largest occupancy of each trajectory. These guards delimit this experiment; they do not establish blanket physical accuracy throughout the enclosed region.

First, three fixed-composition states compare u_T and u_Y with symmetric finite differences at three decreasing step sizes, using independent density inversions. Scaled derivative errors must be below `1e-6` at the smallest step. The transformed differential charge/energy identities must agree within `1e-12` of their declared scales.

Each trajectory is then run at ODE tolerances `2e-6` and `2e-9` with weak quadrature order 96. A third fine-tolerance run uses weak order 160. Shared-time histories and temperature-floor event times must agree within `1e-5` scaled/relative error. Escaped energy must increase and total matter energy decrease; global energy and lepton ledger errors must be below `2e-6`. These are numerical gates of this restricted model, not estimates of omitted nuclear physics.

Run:

```
python3 /Users/rossma/Genesis_LANL/outputs/nmb-finite-t-20261015/thermal_element/cooling_element.py
```

The JSON contains source and dependency SHA-256 hashes checked before and after execution, exact command, constants, library versions, derivative step ladders, run/termination data, sampled histories and refinement errors. If a dependency changes during execution, the script writes a rejected provenance report instead of accepted results. Independent review is required before publishing the new cooling results as accepted campaign evidence.

## Limits that conservation does not remove

An accurate energy ledger does not qualify the static nucleon rates for neutron-star-core density, finite recoil, dense responses or arbitrarily long evolution. It only establishes that the chosen EOS and escaping collision moments have been composed consistently over the tested warm/dilute trajectories. The antineutrino kernel has a real disconnected support band between `Q-me` and `Q+me`; no arbitrary trapped-radiation thermalization claim follows from transparent cooling. A radiation box would need to retain those unconnected occupations or add separately qualified redistribution physics.

Density trajectories, neutrino reabsorption, photon escape, realistic nuclear composition and EOS-consistent interacting rates are subsequent, distinct model extensions.

## Measured prototype results, awaiting independent review

All six trajectory runs pass, alongside three EOS derivative states and four solver/quadrature comparisons. The fine order-160 runs give:

| Initial `(T,nB,Ye)` | Stop time [s] | Final T [MeV] | Final Ye | Escaped energy [MeV/baryon] |
| --- | ---: | ---: | ---: | ---: |
| `(2,1e-7,0.3)` | 0.5830722251 | 1.4 | 0.5285271553 | 29.3544898933 |
| `(5,1e-5,0.5)` | 0.001253491605 | 3.5 | 0.2334007699 | 14.9043099741 |

Across all coarse and fine runs, largest scaled energy and lepton ledger residuals are `4.72e-9` and `5.27e-9`. On the two fine order-160 runs these are each below `3.6e-11`. The largest matched-history solver refinement error is `5.88e-8`; the largest order-96-to160 weak-quadrature history difference is `4.55e-10`. Temperature-floor termination times change by less than `3.1e-11` relatively. The largest actual classical-nucleon occupancy diagnostic is `0.002462`.

These small residuals demonstrate numerical consistency of the stated restricted equations. Their physical systematic uncertainty from static nucleons and omitted interactions is not represented by these residuals. Results retain `publication_status: Requires independent review before publication` until a separate reviewer accepts the equations, tests and source conventions.
