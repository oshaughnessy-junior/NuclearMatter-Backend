# Independent physics review: finite spectral-bin thermal box

15 October 2026. **Accept the declared finite-grid closed-box experiment and selected short-step evidence.** No physics blocker was found. This accepts the specified free-FD matter/current/radiation ansatz, not continuum spectra, grid-converged transport or long-time/stiff thermalization. Author files remain untouched.

Reviewed frozen `thermal_box/box.py` SHA256 `ae356024fd688765f15342d5026f5e9534fcaec96ba713a4e55e25f0658c2710`, author note, study, guards/replay, four/eight-bin LTE results and underlying EOS/current contracts. Original physical-run source and later guarded source are distinguished in the author manifest. We did not rerun the69s author experiment.

## Radiation measure and conserved quantities

For one massless active helicity, the isotropic measure is `4pi p²dp/(2pi hbar)³ = E²dE/[2pi²(hbar c)³]`. Both electron flavors are separate species; there is no extra neutrino spin factor, target spin factor or factor c in `df/dt` when j,chi are s^-1. Integrating this measure over a bin gives the author's w. Its energy moment gives exactly `w Emean`, with `Emean=3(Ehi⁴−Elo⁴)/[4(Ehi³−Elo³)]`.

Thus `sum w f`, `sum w Emean f`, and `sum w h(f)` are exactly the number, energy and occupation entropy of the **bin-constant radiation representation**. Sampling physical coefficients at Emean does not make the projected collision operator an exact integral of the continuum operator. Likewise discrete LTE samples are not exact continuum FD averages. These qualifications are correctly retained.

A positive neutrino collision source gains radiation energy and electron-family lepton number, so material receives `du=−sum w E C dt/nB`, `dYe=−sum sigma w C dt/nB`, sigma=(+1,−1). The code uses these very increments to update radiation and material. A positive antineutrino emission source therefore raises material Ye, with the opposite radiation lepton contribution. All signs agree with the named capture/positron-capture/neutron-decay inventory.

The EOS uses common NR kinetic nucleon mass939MeV, constant neutron Q offset, zero proton offset, full relativistic e± energy and charge neutrality. Subtracting a common baryon rest energy is harmless at fixed baryon number. The physical Q and changing electron/pair energy remain in the inversion; no duplicated Q heating term is inserted. Photons are absent in this experiment. This is the previously qualified free-gas/allowed-current model, without a new interacting or relativistic-medium qualification.

## Entropy identity and finite-step limitation

Let `zeta=mu_e+mu_p−mu_n`. The material first law gives `ds_m/dt=(du/dt−zeta dYe/dt)/T`. Radiation has `h'(f)=log[(1−f)/f]`. Combining the independently signed material sources with radiation entropy gives

`ds_total/dt=sum w C [log((1−f)/f)−(E−sigma zeta)/T]/nB`.

Only after physical KMS `log(j/chi)=−(E−sigma zeta)/T` does this become `sum w(A−B)log(A/B)/nB`, A=j(1−f), B=chi f. The latter is nonnegative for positive A,B. **Checking that inequality alone is insufficient:** arbitrary positive inconsistent coefficients also satisfy it. Our held-out probe explicitly verifies KMS against the EOS and independently reconstructs the material-first-law plus radiation-entropy expression.

The theorem is infinitesimal for this finite-grid system. Finite Euler increments can violate positivity or entropy at larger steps. The code rejects occupations outside[0,1] and checks actual entropy changes rather than clipping them. Source/EOS conservation residuals close to roundoff do not establish temporal accuracy. The author's2/4/8 refinement is useful local evidence, while its stated coarse time/grid and stiff/long-time limits remain open.

## New physical-state and grid witness

`probe.py` chooses a new `T=2.15MeV, Ye=.28, nB=1.3e−4fm^-3`, two irregular bins with edges[0,3,9]MeV, and non-LTE unequal-flavor occupations `[[.25,.07],[.11,.03]]`. It freshly evaluates the named physical channels at this state, then at the changed state after a short conservative step. This is not synthetic-rate or frozen-bath evolution.

**19/19 checks pass in5.33s**, including direct scalar integrals of number/energy DOS, EOS-affinity KMS at all four flavor/energy points, the independent entropy identity, signed material increments, conservation, actual entropy increase, tangent approach, old-state-rate rejection and new-state physical coefficient change.

| Held-out diagnostic | Result |
| --- | ---: |
| Material energy source [MeV baryon^-1 s^-1] | −526.693687 |
| Material Ye source [s^-1] | −78.614279 |
| Total entropy production [baryon^-1 s^-1, kB=1] | +540.854152 |
| Step [s] | 2.82287379e−7 |
| Updated T [MeV] | 2.15016243557 |
| Updated Ye | .279977808181 |
| Actual total entropy increase [baryon^-1] | 1.52363163e−4 |
| Largest physical j/chi relative change after step | .00122752225 |

Both temperatures and compositions changed; the maximum coefficient change is about0.123%. Old-state coefficients are rejected at the new state. Full raw channels, EOS totals, rates, grid and source hashes accompany the probe. These four new energy/state points supplement, rather than broaden into a rectangle, the author source qualification.

## Finite-grid LTE witness

`lte_probe.py` independently reconstructs the author's four-bin LTE totals and verifies its initial conserved E,Le. It then perturbs a neutrino or antineutrino bin in both directions, explicitly compensates material u and Ye to conserve total E and Le, and independently inverts the full EOS. No collision step or imposed radiation reset constructs these perturbations.

**14/14 checks pass in12.63s.** All four constrained perturbations lower total entropy: neutrino changes give approximately−2.39e−6 and−2.54e−6 per baryon; antineutrino changes give−7.37e−9 and−6.89e−9. This is local evidence for the variational equilibrium in tested directions, not a proof of global uniqueness or an integrated approach to LTE. Physical stationarity also follows from the separately verified KMS coefficients and discrete FD occupations.

The eight-bin LTE displacement at equal E,Le is scientifically meaningful grid sensitivity and must remain in published evidence. It does not establish continuum convergence; neither the four-bin short trajectory nor the limited0–24MeV phase space has that qualification. There are no radiation degrees of freedom beyond the specified upper edge. Finite-grid equilibration therefore cannot close an infinite-spectrum tail or redistribution gate.

## Disposition

Accepted: signs, helicity normalization, free-FD energy conventions, the finite-grid H theorem with EOS-consistent KMS, selected physical source feedback, and local conservative/entropy behavior. Continue only with explicit grid/time/source controls. Continuum energy-grid convergence, nontrivial high-energy tails, long-time/stiff evolution, extra interactions and interacting-EOS/current physics remain open. No author gate was loosened, no failure was removed, and no production change was made.
