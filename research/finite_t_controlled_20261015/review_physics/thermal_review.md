# Independent physics review: no-thermostat cooling element

15 October 2026. Reviewed `thermal_element/cooling_element.py`, `TECHNICAL_NOTE.md` and `results.json`. The reviewer did not author this element and changed only `review_physics/`. Campaign quota/account/window checked at start: 3% used, matching the original account/window.

## Disposition

**Accept the two reported warm dilute fixed-density cooling trajectories, their refinement runs, and their qualified physical interpretation. No blocking coupling error was found.** These are self-cooling composition-evolving fluid elements in a declared static free-nucleon ansatz, with trapped/local-equilibrium photons and freely escaping neutrinos. They are not neutron-star-core cooling models, finite-recoil Urca calculations, or independently justified electromagnetic/neutrino transport regimes.

The useful advance is substantive: temperature now follows the same total matter energy that includes reaction composition energy, pairs and photons, without an external thermostat. The independently checked source transformation agrees with this first law. This acceptance concerns the tested trajectories, not all states admitted by the evaluation guard.

## Equation audit

At constant baryon density/volume the first law is `dot(u)=−q_nu`. There is no pressure work. For the stated independent variables `(T,Ye)`,

```
u = (1−Ye)Q + 3T/2 + [e_lepton(T,nB Ye)+a_gamma T^4]/nB
u_T = 3/2 + Cv_lepton|net_charge/nB + 4 a_gamma T^3/nB
u_Y = −Q + mu_e − T mu_T|net_charge
dot(T) = [−q_nu−u_Y F]/u_T.
```

These are the implemented formulas. The neutron/proton rest-energy difference occurs once in `u`; no extra reaction-Q heating appears. The common kinetic mass produces the common `3T/2` term; it does not supply recoil kinematics in the weak kernel. Photon energy uses the correct two-polarization coefficient `a_gamma=pi²/[15(hbar c)³]`. Photons have zero chemical potential and add heat capacity while remaining local: photon luminosity is intentionally absent.

The lepton heat capacity must hold **net** charge fixed, allowing pair creation/destruction as temperature changes. The base evaluator does this. Using a particle-number-frozen derivative would omit pair heat capacity and would not yield the returned energy function's derivative. The lepton composition derivative includes the implicit total chemical-potential response, including rest energy.

For the numerical `(T,mu)` variables, differentiating neutrality yields

`n_T dot(T)+n_mu dot(mu)=nB F`.

Therefore `dot(mu)=[nB F−n_T dot(T)]/n_mu`, as implemented. This is an invertible change of variables at positive susceptibility and heat capacity, not a different composition equation. It does not impose instantaneous beta or kinetic equilibrium.

The escaping lepton ledger is `Delta Ye+N_nue−N_antinue=0`; the energy ledger is `Delta u+E_escaped=0`. Both signs are correct. A decrease in total `u` follows positive emitted energy; monotone temperature is not a general consequence because `u_Y dot(Ye)` may store/release composition energy. The note makes this distinction. The actual runs stop at the first downward crossing of a chosen temperature floor, not at equilibrium.

## Independent targeted checks

`thermal_probes.py` writes `thermal_probe_results.json`. All **8 checks passed**, without replaying the ODE trajectories:

- At both initial states and both reported final states, reconstruct the **full two-variable Jacobian** of returned matter energy and charge by finite differences in `(T,mu)`. Solve this independent 2×2 system with sources `(-q_nu,F)`. Its `dot(T),dot(mu)` agree with the implemented analytical response transformation to at worst **4.86e-10** on the stated scaled norm. This checks the actual coupled response rather than separately repeating its analytic formulas.
- Recompute lepton energy using direct momentum-space adaptive quadrature, combine it with independently written baryon/photon energy, and check each fine-order-160 run's endpoint escaped-energy ledger. Residuals are **1.60e-11 or smaller** on the recorded thermal/exchanged-energy scale. Density inversion and physical constants remain shared with the base reference; this is an independent quadrature/assembly check, not independent many-body theory.
- Bound nucleon degeneracy for **any composition** above each temperature floor by `nB lambda(Tfloor)^3/2`. These conservative upper bounds are **1.26943e-4** and **3.21143e-3**, both below the 0.02 evaluation cutoff. Thus the classical-domain conclusion does not depend solely on sampling a finite number of trajectory points.

The recorded trajectory/refinement results have six runs, three derivative states, and four ODE/quadrature comparisons, with the quoted maximum global scaled energy/lepton residuals **4.72e-9 / 5.27e-9**. The small independent checks support the coupling interpretation of those results; they do not estimate physical systematic uncertainty.

## Domain and termination

The RHS checks finite warm temperature, physical free-nucleon composition, positive frozen-composition heat capacity and a dilute-nucleon occupancy cutoff. The reported floors, **1.4 and 3.5 MeV**, lie well inside the broader `0.20<T<10 MeV` evaluation guard. Both integrations reach an explicit temperature-floor event before the time cap. The termination is a user-chosen experiment boundary and must not be described as freezeout or equilibrium. A generic future solver could reject an ODE trial that overshoots a domain guard before an event is localized; this is a numerical extension concern, not a failure of these interior examples.

Photon trapping/pair equilibrium and neutrino transparency are imposed closures. No parcel size, electromagnetic thermalization rate, or comparison of photon/neutrino optical depths was established. The note already exposes this. It is a legitimate controlled theoretical ansatz, but predictions should retain that conditioning. Likewise, absence of nuclei at these densities/temperatures is a model species restriction; no nuclear statistical equilibrium proof is supplied.

The prior static antineutrino support gap remains. It poses no energy-accounting obstruction when neutrinos immediately escape. It still precludes an unqualified globally thermalizing closed-box claim.

## Next bounded physical increment

A prescribed compression/expansion trajectory can now be built using the same free energy, its pressure, frozen-composition heat capacity and density derivative, adding `P dot(nB)/nB²` to `dot(u)` and the density term to the `(T,mu)` neutrality transformation. First test reversible **weak-off** evolution against conserved entropy and mechanical work, then restore the already reviewed weak moments and both exchange ledgers. This remains a controlled matter element; no momentum equation or full hydro engine is needed. A realistic finite-recoil/blocked interacting weak response remains a separate physics extension.
