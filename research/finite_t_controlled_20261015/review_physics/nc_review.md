# Independent physics review: free neutral-current redistribution

15 October 2026. Reviewed frozen `neutral_current/free_nc.py`, its research note, source manifest, 167-check author result and validation source. Primary sources below were read directly. Reviewer changed only new `review_physics/nc_*` files. Campaign quota/account/window matched at start (5% used).

**Accept the specified free unpolarized leading-current NR kernel, its selected incident-energy benchmarks and the qualitative closed-grid conservation/entropy demonstration.** No angular/prefactor/chemical-reciprocity physics blocker found. **Do not qualify continuum heating/evolution or a stringent general entropy-cancellation tolerance from the closed grid:** a new held-out probe exposes roundoff amplification from large diagonal transitions, preserved and diagnosed below. The external finite-T GM1 gate remains open and unrelated to this free-nucleon acceptance.

## Primary prefactor, couplings and spin convention

[Bedaque, Reddy, Sen and Warrington, Eq.1](https://arxiv.org/html/1801.07077) gives the leading nonrelativistic density/spin differential kernel with GF²/(4pi²), outgoing neutrino energy squared, and vector/axial weights `(1+cos)` / `(3−cos)`. The code follows that normalization with its reviewed energy-delta response. Restoring the fm response measure contributes `(hbarc)³`, and 1/hbar converts the remaining energy rate to seconds. **No CKM factor** applies to this neutral current.

Neutron `CV=−1/2, CA=−gA/2` and proton `CV=1/2−2sin²thetaW, CA=+gA/2` match the declared tree-level convention. Their signs do not affect this static unpolarized symmetric-current calculation, which contains only squares; neutrino/antineutrino differences from weak magnetism or antisymmetric higher-order currents are absent.

The free density correlator and each Cartesian Pauli-spin correlator share the same response because `Tr(sigma_i sigma_j)=2delta_ij`. The imported S already includes the two spin states. The factor three from spatial axial contraction is already in `(3−cos)`; another spin-two or spin-three multiplier would be wrong. Conventions for an externally supplied **interacting** axial structure function must be checked afresh, including whether it is one-component or summed/averaged over components. The cited correlated-medium work is not implemented by setting its density and axial responses equal.

The static total and `(1−cos)` cross sections are independently consistent with [Burrows, Reddy and Thompson](https://arxiv.org/html/astro-ph/0404432):

```
sigma_total = GF² E² (hbarc)²/pi * (CV²+3CA²)
sigma_(1−cos) = 2 GF² E² (hbarc)²/(3pi) * (CV²+5CA²).
```

These use fm²→1e-26 cm² and give the intended bare target cross sections. The separate pure-vector and pure-axial ratios 2/3 and 10/9 test angular weighting. The heavy-common-mass limit is a valid normalization check; it does not include physical recoil-current corrections or certify arbitrary finite-mass momentum domains.

## Energy direction, phase space and reciprocity

For Ein→Eout, `omega=Ein−Eout` is energy **into** baryons and `q²=(Ein−Eout)²+2Ein Eout(1−cos)`. Reversing neutrino energies changes omega sign while keeping q. Neutral diagonal baryon chemical potentials and offsets cancel in the exponential KMS factor but remain in the response amplitude/Pauli blocking.

Because the bare kernel already contains outgoing Eout² phase space,

`K(Eout→Ein)/K(Ein→Eout)=(Ein/Eout)² exp[−(Ein−Eout)/T]`.

The occupation gain integral therefore needs the neutrino density-of-states ratio. The implementation's discrete `gi=Di wi` and outgoing-weighted transition Aij include it correctly; adding another outgoing DOS would double count. Incoming f and outgoing `(1−f)` are each applied once in the pair flux.

FD occupations at bath temperature are stationary for **arbitrary** neutrino chemical potential. Scattering conserves neutrino particle number; it cannot by itself enforce zero chemical potential or create an emission/absorption equilibrium population. The source must remain a redistribution operator rather than be collapsed to a single j/chi relaxation coefficient.

The event `(1−cos)` moment is a meaningful selected angular diagnostic and reduces to the static transport cross section. With inelastic redistribution, it is not a complete transport closure, spectral thermalization rate or universal momentum-opacity scalar. The note correctly restricts that interpretation.

## Material energy and total entropy

For pair flux Jij from bin i→j, the radiation energy source is `sum Ei*dNi`; the bath receives `sum(Ei−Ej)Jij`. They cancel when built from the same transitions. Nuclear composition does not change in these diagonal NC reactions. With a fixed-temperature bath, total entropy production is

`dot(S_rad)+dot(Q_bath)/T = −sum dNi[ln(fi/(1−fi))+Ei/T]`.

Microscopic reciprocity makes this equal to the nonnegative sum over unordered pairs `(Jij−Jji)ln(Jij/Jji)`. This is the relevant entropy statement. Radiation entropy alone need not increase because heat can flow to the external bath. A material EOS energy equation is not evolved here.

## Held-out probes: one preserved cancellation failure

Executed `nc_probes.py`; raw results are in `nc_probe_results.json`, with the initial nine-check failure copied to `nc_initial_probe_failure.json`. The extended probe set has **9/10 passing**, with the same failed strict diagnostic deliberately retained rather than loosened.

The new bath is `T=1.3 MeV, n=.0013 fm^-3`, constant offset 17 MeV, with irregular energy nodes/weights and independently chosen couplings `(CV,CA)=(.2,.7)`. Three arbitrary chemical potentials (−20,7,30 MeV) pass FD stationarity; number conservation and signed bath/radiation energy checks pass. Two additional held-out **proton** reverse-kernel DOS-ratio tests agree within **2.67e-15** logarithmically. Off-equilibrium total entropy production remains positive.

The strict comparison of the entropy computed from the author's returned dNi against the pairwise nonnegative formula misses the reviewer 2e-12 relative target by giving **2.17e-10**. Diagnosis: total transition flux is .03972473035 fm^-3 s^-1 and its diagonal contribution is .03972472058; the nontrivial exchange/entropy signal is many orders smaller. Computing `flux.sum(columns)−flux.sum(rows)` retains huge self-transition terms until the final subtraction and loses digits. The largest dNi discrepancy is only **7.71e-19 fm^-3 s^-1**, consistent with gross-scale floating-point cancellation.

Independently accumulating **off-diagonal antisymmetric pair differences before bin sums**, without any author-file modification, restores the entropy identity to **2.23e-16 relative**. This confirms the detailed-balance physics and locates a numerical-conditioning limitation. It does not repair the frozen implementation by declaration. Before high-precision evolution, preserve source accuracy by omitting identically cancelling self transitions and accumulating pair net fluxes directly; test errors against the actual redistributed energy/entropy signal as well as gross rates.

This failure does not invalidate the author's selected moment checks or entropy-positivity example. It does invalidate interpreting those checks as a universal 1e-12 relative entropy identity on arbitrary closed grids. Exact conservation or equilibrium on a coarse grid also does not establish continuum redistribution accuracy: narrow responses can be missed while these algebraic tests pass.

## Remaining gates

Qualify a resolving energy discretization with independent continuum integrals, tail error and leakage outside any truncated domain, then test non-equilibrium spectral evolution and bath energy exchange with appropriate source scales. Preserve angular dependence for anisotropic transport. Extremely cold/narrow q→0 limits and high-momentum NR tails need explicit treatment; generic radiation argument validation and underflow handling also remain outside the selected-input scope.

No RPA, density/spin correlations, pairing, axial renormalization, relativistic dispersion/current or weak magnetism is qualified. Free neutron and proton kernels can be added consistently only within their declared independent-gas mixture ansatz; the matched EOS factory does not add missing many-body response physics. No source or derivative agreement in this leaf closes the separate external interacting finite-T GM1 benchmark gate.
