# Restricted transparent thermostatted fluid element

15 October 2026. This executable prototype composes the adjacent `thermodynamics/lepton_reference.py` and `weak_rates/static_capture.py` research references. It evolves charge-neutral, dilute classical free neutron/proton matter at prescribed constant baryon density and temperature. It is a controlled composition-relaxation calculation with a heat reservoir, not neutron-star-core cooling or a hydrodynamics solver.

## Equations, species and domain

The matter species are free neutrons/protons and pair-equilibrated ideal electrons/positrons; a photon gas is included as a constant-temperature energy contribution. Charge neutrality gives `n_e,net=Ye*nB`, `n_p=Ye*nB`, `n_n=(1-Ye)*nB`. Captures and neutron decay produce escaping electron neutrinos/antineutrinos with no incident radiation and no neutrino final-state blocking. The directional target moments retain electron mass and reaction Q but omit nucleon recoil, thermal velocities and Pauli blocking, in-medium interactions, pairing, nuclei and muons. This is not a direct-Urca threshold calculation.

Instead of repeatedly solving the electron-density inversion during the ODE, evolve the total electron chemical potential `mu`:

```
F = dYe/dt = (R_positron_capture + R_neutron_decay - R_electron_capture)/nB
Ye(mu) = n_e,net(T,mu)/nB
dmu/dt = nB*F / (dn_e,net/dmu)_T.
```

The ideal-gas susceptibility is strictly positive in the qualified domain. This variable choice preserves the original charge-neutral source equation without an interpolation table. Initial and equilibrium states still use independent monotone root solves.

The three example states are `(T[MeV],nB[fm^-3])=(0.5,1e-8),(2,1e-7),(5,1e-5)`. The report includes a classical-nucleon degeneracy diagnostic `max(n_n,n_p)*lambda_th^3/2`, using a common 939 MeV kinetic mass; it does not claim validity for arbitrary density/temperature.

## Thermostat and energy ledger

Per-baryon matter energy, after removing a constant common proton rest energy, is

```
u = (1-Ye)*Q + 3*T/2 + (epsilon_e+- + epsilon_gamma)/nB.
```

The nucleon kinetic mass is common and the neutron-proton rest offset Q is retained, matching the static weak-reference convention. Constant density and temperature make the classical nucleon kinetic and photon energies fixed. The finite-mass ideal electron/positron energy depends on composition through charge neutrality.

The exact fixed-T derivative follows `d epsilon_e/dmu = mu*d n_e/dmu + T*d n_e/dT`:

```
u_Y = -Q + mu + T*(dn_e/dT)_mu/(dn_e/dmu)_T
H_thermostat/nB = Q_neutrino/nB + u_Y*F.
```

`H_thermostat` is an explicit signed external power supplied to hold T fixed. Its integrated ledger obeys `Delta u = integral(H_thermostat-Q_neutrino)dt/nB`. No extra nuclear Q heat is added: composition-dependent neutron rest energy is already in `u`. Matter energy alone is not conserved because emitted neutrinos escape and the thermostat exchanges heat.

For lepton number, every emitted neutrino and antineutrino is separately accumulated. The closed accounting identity is `Delta Ye + N_nue,escaped - N_antinue,escaped = 0`, with each N expressed per baryon. Total emitted number and energy remain positive at kinetic composition balance; a vanishing net source does not mean the parcel stops cooling if the thermostat is removed.

## Tests and executable provenance

Run:

```
python3 /Users/rossma/Genesis_LANL/outputs/nmb-finite-t-20261015/fluid_element/fixed_temperature_element.py
```

The script writes `results.json` only after all assertions pass. It includes UTC, command, dependency/source SHA-256 values, constants, library versions and small sampled histories. Hashes are checked again at the end to reject results if another worker changed a dependency during the run.

Each case finds the self-consistent transparent kinetic root, checks its net rate against gross directional emission, and verifies positive emissivity. It independently refines weak quadrature from order 96 to 160. The composition derivative in the thermostat equation is checked by finite differences of the declared total matter energy. The ODE integrates from below and above the stable root for twenty local linear relaxation times; a two-tolerance comparison checks numerical sensitivity and monotonic approach. A dilute pair-bath limit is compared with an exactly solvable constant-mu affine source, and pure-target endpoint tests check capture/decay counting signs.

Conservation ledgers are inspected throughout each trajectory, not only at its endpoint. Error denominators use gross exchanged energy or thermal/composition changes, with a declared 1 MeV-per-baryon absolute scale, and gross emitted lepton counts or one lepton per baryon. A constant baryon rest energy is not allowed to hide drift. Measured results are in the JSON; acceptance thresholds in the script are numerical qualifications of these restricted equations.

The ideal lepton and weak references share physical constants and the same chosen reaction theory. Numerical agreement does not establish the accuracy of omitted recoil or in-medium physics. The antineutrino static target channels also have a genuine kinematic support gap `Q-me<E_nu<Q+me`; transparent integrated emission remains meaningful, but these kernels alone cannot thermalize arbitrary occupations in that gap in a closed box. No global thermal-box equilibration claim follows from this fluid-element test.

## Next physical increment

A density trajectory can next evolve `T` using this explicitly shared-mass classical nucleon plus exact lepton/photon EOS, including compression work and composition derivatives. That extension should come after review of these fixed-T ledgers and base-kernel domains. Finite-recoil/blocked dense-matter kernels require a different EOS-consistent response calculation; changing the density history does not supply that physics.

## Measured accepted results

The accepted rerun passes all three cases and twelve ODE runs (two initial compositions and two tolerances for each state). Respective kinetic roots are `Ye=0.5858515076`, `0.5753198402`, and `0.3063052466`. Their total escaping emissivities are strictly positive: `2.78474e-10`, `1.36935e-5`, and `0.314543 MeV fm^-3 s^-1`. The associated relaxation rates are `0.0360891`, `31.2186`, and `4953.36 s^-1`. These are predictions of the declared static classical theory, not universal astrophysical rates.

The largest scaled energy-ledger residual is `2.43e-9`; largest scaled lepton-ledger residual is `5.62e-10`; largest absolute ODE-to-root Ye difference after twenty local relaxation times is `7.88e-10`. The fixed-mu analytic ODE test differs by `1.20e-12`, and the self-consistent dilute limit differs from its pair-bath root by `6.48e-8` in Ye. Classical nucleon occupation diagnostics remain below `0.00131` for the example roots.

Run `check_refinement.py` to reproduce the additional comparison of serialized composition, emitted-energy and thermostat-heat histories. Its six matched histories pass; the largest scaled coarse/fine difference is `9.59e-10`. This separate JSON records hashes of both results and checker.

The first run finished its numerical checks but was rejected when dependency hashes changed during concurrent base-kernel repairs. `initial_failure.json` preserves that event. The accepted rerun began after repairs stabilized and passed the before/after hash gate; the initially rejected run supplied no accepted numerical payload.
