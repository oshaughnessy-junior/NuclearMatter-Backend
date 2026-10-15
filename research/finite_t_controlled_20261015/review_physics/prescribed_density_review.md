# Independent physics review: prescribed-density elements

15 October 2026. Reviewed `prescribed_density/element.py`, its technical note, the ten-run result payload, and the separately formulated entropy-root/work-quadrature oracle. No author files changed. Quota/account/window matched at review start (3% used).

**Accept the reported half-density expansion and double-density compression, with weak reactions disabled and enabled, as controlled research benchmarks of the declared dilute EOS and static weak theory. No blocking thermodynamic or source-transform error found.** They prescribe density rather than solve fluid dynamics. The new result qualifies mechanical-work, dilution and composition accounting; it does not extend weak-rate physical validity to dense matter or establish an electromagnetic/neutrino transport regime.

## Thermodynamic audit

At independent `(T,n,Y)`, the lepton net charge is Yn. Define `A_l=mu−T mu_T|n_l=de_l/dn_l|T`. Differentiating the stated total energy gives

```
u_n|T,Y = [nY A_l−e_l−aT^4]/n²
u_Y|T,n = −Q+A_l
u_T|n,Y = 3/2+Cv_l/n+4aT³/n.
```

The implementation holds the correct variables. In particular, the `nY A_l` contribution would disappear incorrectly if density differentiation held net electron density fixed. The common classical nucleon kinetic energy per baryon has no density dependence at fixed T,Y. Photon energy contributes `−aT^4/n²` to u_n and the usual 4aT³/n heat capacity.

Classical nucleon entropy includes the two-spin thermal wavelength and correct mixing terms. The free energy reproduces `u−Ts`; `P=n² f_n|T,Y` gives `nT+P_l+aT^4/3`. Distinct rest energies affect u and composition affinity but not the frozen-composition entropy/pressure. All audited terms are consistent.

For fixed baryon number with prescribed volume, `d(1/n)/dt=−dot(n)/n²`. Therefore work **on** matter is `P dot(n)/n²`, positive for compression and negative for expansion. The temperature law follows directly:

`u_T dot(T)=[P/n²−u_n]dot(n)−q_nu−u_Y F`.

Charge neutrality differentiates to

`n_T dot(T)+n_mu dot(mu)=nF+Y dot(n)`.

The implemented positive compression/dilution term is essential. These equations preserve Ye when weak interactions are disabled; they do not confuse net electron density with charge per baryon. Neutrino energies and Q are accounted once, with no added reaction-Q heating.

## Independent checks and recorded evidence

`response_density_probes.py` writes `prescribed_density_review_results.json`. For both expansion/compression directions and weak off/on, it reconstructs the **full EOS energy-and-composition Jacobian in `(T,mu,n)` by finite differences**, supplies the prescribed density derivative, and independently solves the 2×2 system for `dot(T),dot(mu)` from the required energy and flavor sources. All four probes pass; largest scaled difference is **3.07e-10**. This tests the density terms together with the coupled source transform without reusing analytical u_n/u_T/u_Y to build the reference Jacobian.

The author's entropy-root oracle is an appropriate independent numerical formulation: with weak reactions off, solve constant specific entropy at each density and separately integrate `P/n*dln(n)`. It does not obtain its temperature path from the ODE RHS. The reported root/ODE agreement and work/EOS endpoint agreement support reversible work handling. This is independence of formulation for the same EOS, not independent interaction theory.

The serialized ten-run suite reports largest scaled energy/lepton residuals **1.70e-12 / 2.91e-12**, weak-off relative entropy drift **1.14e-15**, and maximum classical nucleon occupancy **7.32e-5**. Small errors are credible for these smooth short trajectories and must not be extrapolated to larger excursions. State/output histories and the separately recorded oracle are preferable evidence to test counts alone.

## Conservation and qualifications

The correct energy ledger is `Delta u+E_escaped−W_on_matter=0`. Matter alone is not a closed energy system because the externally prescribed density does work. The flavor ledger remains `Delta Y+N_nue−N_antinue=0` per conserved baryon. Weak-off evolution conserves specific entropy and Y. Weak-on matter can emit entropy into escaping neutrinos, so matter entropy alone must not be required to increase or remain constant; the implementation correctly gates entropy conservation only with weak processes disabled.

The 0.8 MeV event is a protective experiment boundary; accepted examples finish at their prescribed 0.1 s density endpoint without reaching it. The warm-temperature, composition and classical-occupancy guards restrict evaluation but do not imply blanket physical qualification throughout the admitted region. Trial-step domain handling may need stronger event/recovery logic for larger excursions, especially around composition endpoints.

Local photon retention, electromagnetic/pair equilibrium, free nucleons and neutrino escape remain imposed physical ansatz. Nuclei, finite-recoil/blocked weak responses, interactions and muons remain absent. The scalar recoil/Pauli response developed in parallel has not yet been contracted into these rates and must not be described as already incorporated. The static antineutrino gap remains harmless to escaped-energy bookkeeping and still prevents unrestricted closed-box equilibration claims.

## Recommended next connection

Retain these elements as numerical contract tests when introducing the separately reviewed dynamical response. First qualify its lepton-current contraction and optical normalization at fixed state. Then replace the static kernel in one identical prescribed-density trajectory, reporting the change as physical-model dependence rather than numerical error. Pressure/energy derivatives must continue to come from a compatible EOS. A prescribed slab/annulus irradiation example is another useful controlled increment after the same source and reservoir conventions are preserved; a momentum equation or full GRMHD engine is unnecessary.
