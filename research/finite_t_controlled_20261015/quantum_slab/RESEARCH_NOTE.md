# Irradiated quantum fixed-bath slab: selected spectral energies

15 October 2026. This integration leaf composes the matched free-FD nucleon/lepton EOS with capture/inverse and crossed neutron-decay/inverse optical coefficients, then solves a steady plane slab with prescribed incoming radiation. It retains the leading allowed-current, common-mass nonrelativistic ansatz. **This is a selected-energy transfer checkpoint, not an integrated luminosity, a relativistic/GM1 opacity, a coupled material evolution or a production backend.** Independent review of the mapped-capture extension and this integration remains required.

## Reproduction and scope

```bash
python outputs/nmb-finite-t-20261015/quantum_slab/validate_slab.py
python outputs/nmb-finite-t-20261015/quantum_slab/check_guard.py
```

The main suite passes **506/506** checks in about 5.3 seconds. The separate panel-guard comparison passes **8/8** checks. Full measured spectral coefficients, state diagnostics, incoming/outgoing fluxes, material-source accounting and optical-depth ladders are saved in `validation_results.json`. `guard_validation.json` records the numerical extension described below.

Two independent-composition matter states are used, both with T=2 MeV and common kinetic nucleon mass 939 MeV:

| State | nn [fm^-3] | np [fm^-3] | Selected nu_e energies [MeV] | Selected anti-nu_e energies [MeV] |
|---|---:|---:|---|---|
| Dilute nucleons | 7e-5 | 3e-5 | 2,10,20 | 0.4,0.85,4 |
| Moderately FD nucleons | 0.002 | 0.001 | 20,40,60 | 0.4,0.85,4 |

`nr_baryon_eos/free_nucleons.py` supplies both nucleon chemical potentials and the same net electron density as np. Neutron/proton constant energy offsets are `(Q,0)` after a common baryon rest-energy subtraction. Their FD occupations, susceptibilities and recoil/Pauli optical response therefore refer to one matched quadratic dispersion. The JSON records pF/M, T/M, chemical potentials and EOS states; no classical baryon energy is substituted.

No full neutrino-energy integration was attempted in this bounded slice. Per-energy fluxes have units cm^-2 s^-1 MeV^-1; multiplying by E gives MeV cm^-2 s^-1 MeV^-1. A twelve-point table must not be mistaken for a converged spectrum or quadrature for a total luminosity.

## Named reaction inventory and chemical signs

For electron neutrinos, the capture reference supplies `p+e- -> n+nu_e` and inverse `nu_e+n -> p+e-`. For antineutrinos, the sum includes `n+e+ -> p+anti-nu_e` and inverse, plus `n -> p+e-+anti-nu_e` and its three-body inverse. The crossed channel uses its own sum-momentum/sum-energy transfer, not the capture kinematics. Both antineutrino constituents are calculated separately before summing in logarithms.

The matter factories imply

```
mu_nue=mu_e+mu_p-mu_n,
mu_antinue=-mu_nue.
```

Each coefficient pair is calculated directly and must pass `log(j/chi)=-(E-mu_nu)/T`; the sum must pass independently. KMS alone does not establish quadrature convergence, because its algebraic factor can cancel even in an inaccurate common integration. A separate 48/48 to 72/72 energy/angle refinement and lepton-tail change 40 T to 50 T is therefore tested at every selected energy.

The collision convention is

```
C[f]=j(1-f)-chi*f=(j+chi)(f_eq-f),
f_eq=j/(j+chi).
```

j and true chi have units s^-1. True absorption is `chi/c`, whereas fixed-bath relaxation/attenuation is `(j+chi)/c`. Scattering, pair reactions, redistribution, oscillations, current recoil/weak magnetism, nuclear interactions, pairing and clusters are absent. No unrestricted FD radiation equilibrium in a closed evolving system is claimed.

## Exact ray and two-face flux convention

For slab thickness L and signed direction cosine mu, define `kappa=(j+chi)/c`, `tau=kappa L`, `t=kappa x`. Incoming hemispheres are isotropic but different:

```
f_left(E)=FD(E,T=4MeV,mu=3MeV),
f_right(E)=0.35 FD(E,T=6MeV,mu=-2MeV).
```

For mu>0 use incoming f_left and optical distance t; for mu<0 use f_right and distance tau-t. The solution is the positive sum

```
f=f_eq*(1-exp(-distance/|mu|))+f_in*exp(-distance/|mu|).
```

It preserves 0<=f<=1 without clipping. Let `D=1/[2pi²(hbar*c)³]` for **one neutrino helicity**, `A=c D E²*10^39`, and `E3(tau)=integral_0^1 mu exp(-tau/mu)dmu`. Number-flux magnitudes are

```
F_in_left=A*f_left/4,       F_in_right=A*f_right/4
F_out_left=A/2*[f_eq*(1/2-E3)+f_right*E3]
F_out_right=A/2*[f_eq*(1/2-E3)+f_left*E3]
F_gain=A*[f_eq-(f_left+f_right)/2]*(1/2-E3).
```

All outgoing quantities are positive outward magnitudes at their respective faces. `F_gain=F_out_left+F_out_right-F_in_left-F_in_right` can have either sign. Energy and per-radiation-species number exchange with the external bath are recorded with opposite signs. Electron-lepton number separately weights nu_e by +1 and anti-nu_e by -1; the bath electron-lepton source is the negative of that signed radiation gain, and is checked explicitly. The generic bath-number counterterm is not a claim that total neutrino particle number is a conserved matter variable. This is conservation with an externally maintained bath, not an integrated EOS evolution or a lepton-number-conserving closed box.

Independent angular QUADPACK integrals of the ray occupations reproduce both face fluxes. An independently integrated depth source uses the angular average `E2(t)` on explicit boundary-layer panels and matches the incoming/outgoing difference. A direct ray ODE using `j(1-f)-chi*f` matches the analytic solution. Optical depths `(1e-6,0.03,1,30,1000)` check thin and opaque behavior at every energy; additional tables use a common physical thickness L=1 km.

A one-face isotropic equilibrium flux is `c D E² f_eq/4`; both faces together carry twice that. Independent numerical FD integrals at zero neutrino chemical potential reproduce the one-helicity, one-face blackbody number and energy formulas. Heavy-mass tests at M=1e7 MeV recover the static target normalization for electron capture, low-energy neutron decay and positron capture, with no fitted prefactor. M is changed only for that mathematical limit.

## A useful true-versus-relaxation example

At the moderately FD state and E_nue=20 MeV, measured coefficients are approximately

```
j=9.55108e5 s^-1,
chi=6.39621e-3 s^-1,
f_eq=0.9999999933,
chi/(j+chi)=6.69685e-9.
```

The 1 km slab has relaxation optical depth 3.18590, while true absorption optical depth is only about 2.13e-8. Replacing relaxation attenuation by true absorption would therefore give a very different fixed-bath transfer solution. This is the Pauli-blocked emission/absorption convention, not a newly invented scattering opacity.

At E_antinue=0.85 MeV, inside the former static gap, finite-T crossed decay gives a nonzero response. Its relaxation depth in the 1 km slab is 9.1920e-7 for dilute nucleons and 3.4361e-5 for the moderately FD state. Mathematical support does not imply rapid equilibration; neither value establishes meaningful spectral redistribution.

## Preserved failures and numerical repairs

The frozen mapped-capture refinement originally failed at the moderately FD state, E_nue=60 MeV, order 72 and tail 50. Source energy edges 61.293332359999994 and 61.29333236 map to the **same float64** `sqrt(Ee-me)=7.796302547317542`. Their transformed quadrature weight is zero, causing `log(0)`. This is not a physical lepton endpoint: p_e is 61.29 MeV. Exact replay and diagnostic data are preserved in `initial_frozen_endpoint_failure.json` and `endpoint_diagnostic.json`.

`guarded_capture.py` is a clearly labelled leaf-local numerical extension. It uses the same equations, response, prefactor and panel construction, discarding only panels whose transformed endpoints are identical. It also evaluates the algebraically equivalent momentum `p=x*sqrt(2me+x²)` rather than subtracting nearly equal squared energies. Frozen files are unchanged. The wrapper uses this guard only after the original failure and requires a recorded collapsed panel; all other errors propagate. Tests compare it with the original evaluator on representable cases. This guard and its integration require independent review; it is not silently attributed to the previously reviewed source.

The initial transfer run passed 427/446 checks and is preserved in `initial_transfer_failures.json`. My first outgoing-flux expression subtracted large incoming/emission terms in the opaque, tiny-antineutrino-occupation limit; the independent angular reference also used an absolute tolerance too loose for those small face occupations. Positive emission-plus-transmission sums remove the cancellation. Scaling the independent angular absolute tolerance to actual face occupation resolves the reference error, including thin grazing rays. No physical rates, prefactors or acceptance tolerances were changed to force agreement.

## Limits and next gate

The matter is nonrelativistic by declared ansatz, with thermal motion, recoil phase space and Pauli blocking but only leading static allowed-current vertices. High-momentum NR tails are formal model behavior, not relativistic predictions. The 0.85 MeV sample probes just above the static decay endpoint; more distant forbidden-region tails may require nucleon momenta outside NR validity. The source-level matched-EOS qualification does not justify arbitrary density, energy or offset choices.

Next independently review the local panel guard, flavor/channel composition and absolute two-face normalization. Only then extend to a neutrino-energy grid whose values, thresholds and tails converge for both chemical-potential signs, with incoming, outgoing and material-source integrals checked separately before taking a potentially cancelling net difference. A closed material/radiation evolution additionally needs matched collision-energy/composition moments and an EOS energy equation; this fixed-bath solution does not supply them.
