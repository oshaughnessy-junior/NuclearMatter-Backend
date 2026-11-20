# Closed finite spectral-bin matter–neutrino thermal box

Research prototype,15October2026. This is a homogeneous fixed-baryon-density **closed finite radiation system**, with physical charged-current coefficients and matched free-NR FD n,p,e± matter. It is not a continuum spectral rate calculation or a hydro engine. Independent review is required. The original physical experiment passes83 gates in69.14s; separate rate-contract guards, valid-state replay and an eight-bin LTE comparison are documented below.

## Physical and spectral ansatz

Matter starts atnB=1e-4fm^-3,T=2MeV,Ye=.3, with no photons. Neutrons/protons use common939MeV kinetic mass and a fixed physicalQ neutron offset; common baryon rest energy is subtracted while neutronQ and total electron/positron energy remain. Electrons obey pair equilibrium and neutralityne,net=np. EOS pressure/entropy/energy and energy inversion are genuine FD values. There are no interactions, nuclei, muons, pairing, weak-magnetism/relativistic-current corrections, scattering or spatial escape.

Radiation has explicitnue andantinue, one helicity each, isotropic homogeneous occupations. Four bins per flavor cover edges[0,4,8,14,24]MeV. ForD=1/[2pi²(hbarc)³], each bin has

    w_i = D (E_hi³−E_lo³)/3,
    E_i = 3(E_hi⁴−E_lo⁴)/[4(E_hi³−E_lo³)].

The occupationf_ai is constant within a bin. ConsequentlyNrad=sumw_i f_ai andUrad=sumw_i E_i f_ai exactly integrate the number/energy of this **piecewise constant representation**. Bin energyE_i is the phase-weighted mean, not an independently chosen endpoint. There are no degrees of freedom outside0–24MeV. This finite phase-space restriction is an ansatz, not a tested continuum high-energy cutoff.

Eachj_ai,chi_ai is the accepted physical allowed-current free-NR capture coefficient evaluated atE_i: electron capture/inverse fornue, and positron capture plus crossed neutron decay/inverses forantinue, explicitly through guarded mapped capture and crossed decay. All use the current materialT,nB,Ye. State/grid-identical initial calls may be shared across temporal resolutions; every distinct material state recomputes actual physical coefficients. No fixed-rate evolving-material approximation is made. Internal quadrature is48/tail40; a selected endpoint energy for each flavor is checked at64/tail50.

Initial radiation hasf_nue=f_antinue=FD(E_i,Tnu=3MeV,munu=0); its net electron-lepton number is zero. These occupations are sampled within the finite-bin ansatz, not claimed as exact continuum FD bin averages.

## Closed exchange and entropy

Let sigma_nue=+1,sigma_antinue=−1. The finite system is

    df_ai/dt = C_ai = j_ai(1−f_ai)−chi_ai f_ai,
    du/dt = −sum_ai w_i E_i C_ai/nB,
    dYe/dt = −sum_ai sigma_a w_i C_ai/nB.

There is no extra factorc or charged-lepton spin factor in these weights. Radiation energy and electron-family lepton number enter exactly the same increments used to update the material. Thus

    Etotal = u + sumw_i E_i f_ai/nB,
    Ltotal = Ye + sumsigma_a w_i f_ai/nB

are conserved by the discrete bookkeeping. Material temperature is recovered from the full FD EOS at the updatedu,Ye. This avoids interpreting composition energy as a prescribed heat capacity. The algorithm uses explicit Euler forf, then matching conservativeu/Ye increments and FD inversion. It rejects occupations leaving[0,1]; no clipping or rate floor enforces positivity. Small selected steps, not a general positivity/entropy theorem for finite Euler steps, justify this run.

Withh(f)=−f logf−(1−f)log(1−f), total entropy per baryon is

    stotal = smatter + sumw_i h(f_ai)/nB.

Here smatter includes all three material species, and electron pairs, from the full FD EOS. Its first law isT dsmatter=du−zeta dYe, wherezeta=mu_e+mu_p−mu_n. Physical KMS giveslog(j/chi)=−(E_i−sigma_a zeta)/T. Therefore the **infinitesimal finite-grid** entropy production is

    dstotal/dt = sumw_i [A_ai−B_ai] log(A_ai/B_ai)/nB >= 0,
    A=j(1−f), B=chi f.

The code evaluates forward and reverse coefficients separately and checks this expression. It also checks actual finite-step entropy changes, because the infinitesimal theorem alone does not certify the coupled Euler update.

## Finite-grid LTE and grid sensitivity

Stationarity in this ansatz isf_ai=FD((E_i−sigma_a zeta)/T). Solve the two **conserved totalE,L constraints** simultaneously for materialT,Ye and these finite-grid occupations; this solve does not integrate the dynamics. Physicalj/chi are freshly evaluated at that different equilibrium matter state and verify collision stationarity. No scalar opacity shortcut or forced change to the evolving radiation is used.

For the four-bin system, initialEtotal=8.55408511538626MeV/baryon andLtotal=.3. The constrained LTE solution isT=2.20479992122230MeV,Ye=.245799344183679. Entropy is6.17665646259175 per baryon, exceeding the initial6.04274998496916. This is a stationary solution in the named finite phase space; one nonlinear root solve is not an exhaustive global state/branch certificate.

A separate eight-bin LTE diagnostic uses edges[0,2,4,6,8,11,14,19,24] at the **same totalE,L**, and checks its freshly computed physical collision term. All3 gates pass in10.21s. It shiftsT by+.0121722188MeV andYe by−.00393998019. This visible grid sensitivity is retained. Neither the LTE state nor the short trajectory is claimed energy-grid converged, and no continuum relaxation time is inferred.

## Short coupled experiment and time sensitivity

Set duration=.08/max_initial(j+chi)=1.13310548449e-6s and compare2/4/8Euler steps. The largest observed dt(j+chi) is.04012. The eight-step endpoint isT=2.00201681750858MeV,Ye=.297534968365090, with both flavors' occupations strictly inside(0,1). It is far from the independently solved LTE state; finite tails do not demonstrate rapid thermalization.

All14 recorded steps preserve total energy and electron-lepton number on their named gates, increase total entropy, and have nonnegative instantaneous entropy production. Final entropy is6.05241662153472 per baryon. Material entropy alone decreases6.01956872→6.01756921 while radiation entropy increases. Testing only the material entropy would be incorrect for this closed exchange.

The2→4 versus4→8 endpoint differences are:

| Quantity | 2minus4 magnitude | 4minus8 magnitude |
|---|---:|---:|
| T [MeV] |8.72921e-6|4.17707e-6|
| Ye |2.57528e-5|1.25677e-5|
| maximum occupation difference |9.48295e-5|4.62143e-5|

These decreasing differences support local first-order sensitivity, not an exact trajectory-error bound, stiff stability certificate or long-time relaxation result. Algebraically exact conserved ledgers do not prove time accuracy. No energy-grid-refined trajectory is run.

## Evidence, guards and limits

`study.py` saves each completed temporal resolution and a partial checkpoint on failure; the experiment has a105s runtime bound. Thirteen distinct material states received physical source evaluations, including the independently solved LTE state. The successful original83-gate run and exact source are preserved as `initial83pass_before_rate_guards.json` and `initial_box_before_rate_guards.py`; `results.json` retains that original run and its source hashes.

Afterwards the leaf added strict currentT,Ye,nB/grid and finite-positive-rate validation to the step/entropy interfaces. This introduces no new physical calculation. `check_controls.py` passes20 invalid-input/exact-valid-state checks. `replay_guarded.py` checks every recorded step again with its **exact original state/grid physical coefficients**, preserving all outputs; it is explicitly a guard regression rather than a fresh changing-state source experiment. No old-state rate is used at a new state. Upstream sources remain unchanged. Current and preserved source hashes are separately manifest in `CHECKPOINT.json`.

No entropy, positivity, temporal-refinement or physical-state gate failed in the bounded run; no failed gate was discarded or loosened. Independent review should reconstruct the DOS/entropy and signed energy/lepton increments, verify finite-grid equilibrium, and sample physical sources at a held-out matter state. General rate/EOS domain coverage, continuum spectral convergence, more complete reaction inventories and long-time/stiff integration remain open.
