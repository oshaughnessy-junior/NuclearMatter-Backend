# Resolving the selected cold capture quadrature failure

This leaf supplies a separately reviewable numerical integration of the **same** physical free-NR allowed-current capture response. It repairs no frozen module. The oldT=0.01MeV threshold extension remains failed in `urca_threshold/initial_cold_extension_failure.json`, with its source snapshot intact. The new result requires independent review before promotion.

## Selected physical state and conventions

Use the threshold experiment's local constant-offset common-mass quasiparticle toy: nB=0.01fm^-3,Yp=0.1,M=939MeV, electron-only neutrality. AtT=0 the externally prescribed neutron-minus-proton offset is calibrated once to mu_n=mu_p+mu_e and is then held fixed. This is not an interacting EOS or a composition derivative across separately calibrated Hamiltonians. Warm neutron/proton FD density roots and the relativistic electron/positron neutrality root are checked first against momentum-coordinate integrals using the previously reviewed threshold helper.

StudyT=(0.01,0.03,0.007)MeV withE_nu=3T. The first is the exact failed point, the second a previously qualified core point, and the third a held-out colder point. These are finite incident neutrino energies; their behavior must not be identified with theE_nu→0 cold triangle limit. No Urca thermal smoothing width or surrogate is introduced.

For absorption the ordered nuclear pair is n→p, omega=E_nu−E_e and q²=E_nu²+p_e²−2E_nu p_e c. With K and the leading allowed angular factor W inherited from the reviewed capture normalization,

    chi = K integral_me^infinity dEe Ee pe [1-fe(Ee)] integral_-1^1 dc W(Ee,c) S_np(q,omega)
    j   = K integral_me^infinity dEe Ee pe fe(Ee)     integral_-1^1 dc W(Ee,c) S_pn(q,-omega).

Both coefficients are independently integrated, in s^-1. KMS is checked afterwards; it is never used to construct one coefficient from the other. Rest/offset chemical potentials follow the frozen response and lepton conventions. The current/response remains leading free nonrelativistic, without interactions, correlations, modified Urca, weak magnetism or relativistic weak-current corrections.

## Two numerical coordinates

`adaptive_yq` is an extension of the **former threshold independent-reference method**, not new independent physical validation. It integrates y=(Ee−mu_e)/T adaptively with QUADPACK. Inner Gauss quadrature usesq∈[|E_nu−pe|,E_nu+pe]. The exact Jacobian changes pe dc into(q/E_nu)dq. Electron-Fermi panels and additional negative-y panels resolve activated contributions below the electron chemical potential. Each coefficient has its own sampled logarithmic scale before adaptive integration; the result remains a logarithm until an optional ordinary representation is requested.

`gauss_Ec` uses a different integration order and coordinate: outer Gauss in cosine, inner direct electron-energy Gauss panels. For each cosine it finds the roots of the two zero-temperature response branch conditions

    Emin(q,omega)=nu_n,
    Emin(q,omega)+w=nu_p,   w=omega−(offset_p−offset_n).

Here Emin=[M w/q−q/2]²/(2M). The roots and nearbyT-spaced locations are **integration panel edges only**. The finite-T analytic response is evaluated unmodified on both sides; neither its support nor its width is replaced by these cold conditions. Electron-Fermi panels resolve the occupation factors separately. Logsumexp accumulates weighted terms for j and chi independently.

The electron lower endpoint is physicalme and remains included; no low-energy support cut is imposed. Upper energy is max(me,mu_e)+tail*T. Tail100→160 is an explicit numerical cutoff test, not a proof of an infinite-range remainder bound. Each method covers the entire chosen finite interval. The adaptive method rejects a q interval collapsed by float64 resolution rather than returning a false zero.

The interface deliberately requiresmu_e>me for this selected degenerate-electron domain. Finite positiveE_nu, integer orders>=8, tail60..240 and bounded adaptive tolerances are enforced. q=0 as a response distribution remains outside the underlying kernel; interior angular nodes do not evaluate that endpoint. A finite logarithm is retained when its ordinary exponential is unavailable. No positivity floor is applied.

## Results and evidence

| T [MeV] | E_nu [MeV] | j [s^-1] | chi [s^-1] |
|---|---|---:|---:|
| 0.01 | 0.03 | 1.42714116630e-13 | 2.87718968204e-12 |
| 0.03 | 0.09 | 2.51293345650e-2 | 5.10411423454e-1 |
| 0.007 | 0.021 | 1.68512943225e-20 | 3.39351181976e-19 |

All these ordinary values are representable; `results.json` also records the logarithms and flags. The main study passes47 gates: densities first, separate j/chi coordinate and refinement comparisons at the unchanged3e-6 gate, adaptive error estimates, KMS and comparison to the unchanged qualifiedT=.03 core. The former failedT=.01 mapped48 and mapped72 results remain discrepant by approximately0.698% and0.0360% from the resolved result. Original48→72 failure0.6623% is retained unchanged. The new coordinate agreement is about4e-14 at that point, which is numerical consistency within a shared physical response, not a statement of physical-model accuracy.

`refine_controls.py` additionally varies order, angular order and upper tail individually at the exact failed point. `check_controls.py` supplies30 invalid-input/representation checks, including a rejected collapsed-q interval and positive IEEE subnormal representation. `results.json` records all methods/controls, roots' input state, chemical potentials and source hashes. Existing frozen failure/source hashes are verified before and after the study.

Reproduce `python3 outputs/nmb-finite-t-20261015/cold_capture/study.py`, `refine_controls.py`, and `check_controls.py`. The two main runs cost approximately3.47s and3.54s; initial45-gate evidence before API guard hardening is preserved separately. No main-study physics gate failed. This small selected-point qualification does not establish a low-T density rectangle, a general optical API, full thermal Urca emissivity or a fit. Independent review should vary root panels/angular integration and check the tail/scale choices before downstream adoption.
