# Actual finite-T GM1 threshold probes with independently classified cold windows

**Author-frozen research prototype; independent review required.** Below the cold electron-Urca triangle onset, this pilot obtains a positive finite-T differential response where the exact cold particle-hole occupation window is empty at identical incident/lepton kinematics. The result comes from physical FD occupations and the declared full current, without a fitted threshold or smearing width. At and above the triangle onset, the chosen finite incident neutrino energy already opens a cold window; those cases are explicitly not called thermal opening.

## Model, states and kinematics

The reviewed internal common939MeV no-sea GM1 Hartree reference gives the cold zero-neutrino-momentum electron triangle density `ncrit=.2780845280467523fm^-3`. At each density factor .97,1,1.03, the cold charge-neutral beta solution sets independent Ye,Ymu and Yp=Ye+Ymu. Those **fractions are held fixed** as temperature changes through .1,.3,1MeV. They are not re-solved warm beta-equilibrium fractions.

For each density, the physical incident energy is fixed to .3MeV, the outgoing total electron energy to that density's cold shared lepton chemical potential, and the lepton scattering cosine to−1. Every one of these physical kinematic quantities stays unchanged across temperature and field closure. El differs across densities because their cold mu_e differs. A held-out witness uses cosine−.43, classified separately at its own physical kinematics.

Two closures are kept separate:

1. **Frozen cold fields:** keep the cold Dirac mass and neutron/proton vector potentials. Independently invert positive-T free quasiparticle densities for both kinetic chemical potentials, then add the prescribed vector shifts. Independently invert both net pair-lepton densities. This is a prescribed cold Hamiltonian with thermal populations, not a self-consistent thermal Hartree bulk EOS or energy ledger.
2. **Self-consistent warm fields:** solve the accepted full finite-T GM1 factory at the same density and fixed cold fractions. Its complete field/root, local stability, neutrality, energy and antibaryon diagnostics are retained.

The response follows the declared [Guo2020 current, Eqs23–28](https://arxiv.org/html/2006.12051v2): kinetic spinors use mD, total energies add Ui, kinetic transfer is `w=omega−Ub+Ua`, scalar form factors use physical `q²=omega²−qvec²`, and WM/PS vertices use kinetic q*. Their weak-current normalization remains the bare939MeV. No physical vacuum neutron-proton Q, ad hoc additional potential shift, RPA, conserved-tensor component reduction or correlation correction is added. The full16-complex-component W and separate direct emission/absorption occupations are retained.

## Exact cold reference at the actual finite incident energy

For equal positive Dirac masses M and spacelike kinetic transfer, direct on-shell kinematics give

`Emin = −w/2 + q/2 sqrt[1+4M²/(q²−w²)]`.

At the same density and Hamiltonian let `nu_nF=sqrt(M²+pFn²)` and `nu_pF=sqrt(M²+pFp²)`. The **cold baryon absorption** window is

`max(Emin,nu_pF−w) < Ea < nu_nF`.

The inverse-emission window is `max(Emin,nu_nF) < Ea < nu_pF−w`. These exact energy/occupation conditions are stricter than looking at a zero-neutrino Fermi triangle alone. An independent Brent root of the original shell-cosine boundary reproduces Emin; open windows have explicit on-shell, total-energy-conserving occupation witnesses. The outgoing electron lies exactly on its cold Fermi surface: a differential T0 boundary value convention1/2 can be used, while a single surface value does not define integrated phase volume. No T0 integrated rate is computed.

At .97 ncrit the cold absorption signed gap is **−3.02858676336848MeV**, so the window is empty. At1 and1.03 ncrit it is already open with width .3MeV for these kinematics. The independently bracketed cold opening of this **selected finite-Enu spectral probe**, with El following cold mu_e along density, is `.2758839462423991fm^-3`, below the zero-neutrino triangle density. This is not asserted to be an onset of the full lepton-integrated rate.

For every self-consistent warm Hamiltonian I also fill the same density-defined cold Fermi spheres using its actual instantaneous mD and Ui and reclassify the cold window. This is a counterfactual occupation diagnostic at fixed fields, not another self-consistent cold EOS. The below-onset window remains empty for all three temperatures; its gap runs from−3.028633 to−3.033236MeV. Therefore those warm-field responses also require thermal occupations rather than a newly cold-open instantaneous Hamiltonian. At and above ncrit both closures are already cold-open at finite Enu. All cold beta-reference inverse-emission windows are empty for positive Enu.

## Computed coefficients, not widths or total Urca rates

The table shows the below-onset density .97 ncrit. Units of chi and j are `s^-1 MeV^-1 per unit lepton cosine`. Chi is true differential absorption, j is inverse emission, and neither is a volume emissivity or an integrated opacity.

| T [MeV] | Frozen-field chi | Warm-field chi | Frozen-field j | Frozen-field zeta_e [MeV] |
|---:|---:|---:|---:|---:|
|.1|3.031353857e-9|3.029945521e-9|1.503974624e-10|−.000348307744|
|.3|5.340096279|5.332673013|1.944089882|−.00313492266|
|1|20061.85861|19972.85617|14353.13689|−.0348519142|

All18 state/closure combinations retain their j,chi,finite logs, representability flags and full W. No ordinary underflow occurs at these selected points, no rate floor is used, and there is no channel dropped to avoid an unrepresentable inverse coefficient. Cold support and positivity follow separate physical tests. Both electron and muon affinities `zeta_l=mu_l+mu_p−mu_n` are recorded; their nonzero values are not warm beta equilibrium. KMS uses them after independently integrating the two occupation products, not as a way to manufacture j from chi.

The figure `selected_thermal_probes.png`/`.pdf` shows the actual three temperature points and the closure comparison. Warm fields reduce the below-onset chi by about.0465%,.1390%,.4436% at these temperatures. That small selected closure difference is not a nuclear/current uncertainty estimate. Joining lines are visual guides, not fitted activation laws or broadening widths. The figure was visually inspected.

## Independent numerical and analytic witnesses

The main18 states pass196 checks in4.79s. Response energy order24→40 and tail50T→80T change log coefficients by at most1.21e-13 under the unchanged3e-7 log gate. Independent direct momentum integrals recover baryon and net pair-lepton densities; separate μ roots in that momentum coordinate reproduce selected EOS roots within1.71e-12MeV. This checks actual density matching rather than assuming a classical baryon gas or fixed chemical potential.

`heldout.py` adds44 checks at the below-onset density, T=.1 and1MeV, both closures, and interior cosine−.43. It reuses the previously frozen independent cylindrical original-delta recipe and independently reconstructed source vertex/spinors; only that pure function is loaded, so no other review file is executed or overwritten. The original total-energy delta is solved numerically. Both j/chi agree with the new q-axis calculation within5.67e-13 relative, and the full W and original-shell residuals/refinement pass. Its own cold window is empty. This is a reused independent **method** author cross-check, not a fresh independent review or external software calculation. An initial missing-global scope error in this helper and an initial main-driver syntax error are preserved; neither changed physics or a test tolerance.

`boltzmann_limit.py` adds16 checks without numerical energy quadrature. At fixed transfer, the azimuth-averaged contracted full-current trace is exactly quadratic in initial energy: `F(Emin+x)=F0+B x+A x²`. Coefficients are algebraically reconstructed and checked against an additional exact trace point. The Boltzmann initial-tail integral is therefore

`exp[(nu_n−Emin)/T] [F0 T+B T²+2 A T³]`.

It retains the actual field/mass/chemical roots, exact electron blocking and absolute weak prefactor. This is an analytic limit of the same physical chain, not a replacement surrogate or a fitted thermal width. With `za=exp[(nu_n−Emin)/T]` and `zb=exp[(nu_p−Emin−w)/T]`, positivity bounds the exact FD-to-Boltzmann coefficient ratio by

`1/[(1+za)(1+zb)] <= chi_FD/chi_MB <= 1`.

At T.1 the analytic result matches within about1e-12 numerical roundoff; the initial neutron activation energy is3.0286863MeV in frozen fields and3.0287328MeV in warm fields. At T.3 the computed ratios are .9999719016/.9999719368, inside their independently evaluated intervals [.9999437857,1]/[.9999438561,1]. Thus the tiny sub-onset response is consistent with a thermally activated population beyond an independently empty cold window and with the absolute physical kernel normalization. This check is not an empirical fit to the rate values.

All **256 author checks pass**. The main experiment and all additional probes are far below120s. Sources, full states, raw numerical evidence and environment versions are pinned in `CHECKPOINT.json`. No frozen upstream, production or publication file was edited.

## Limits and next gates

This establishes selected differential electron-neutrino capture/inverse responses only. It does not compute a lepton-integrated Urca emissivity, complete antineutrino inventory/crossed decay, transport moment, finite-T threshold width, muon-sector rate, material evolution or derivative. The GM1/cold source convention remains a common-mass rounded-coupling phenomenological Hartree ansatz. The external matched finite-T GM1 forward gate, global phase selection, pairing/correlations and interacting-current/model uncertainty remain open.

Independent review should reproduce a below-onset cold window and density-matched finite-T coefficient before extending to spectral/incident-energy integrals. A warm beta-equilibrium or prescribed nonequilibrium study must explicitly choose and solve its intended composition/neutrino constraints instead of relabeling these fixed cold fractions as equilibrium.

Reproduction uses the existing base runtime, without installs:

```bash
/Users/rossma/miniconda3/bin/python probe.py
/Users/rossma/miniconda3/bin/python heldout.py
/Users/rossma/miniconda3/bin/python boltzmann_limit.py
MPLCONFIGDIR=/tmp/nmb-thermal-figure-cache /Users/rossma/miniconda3/bin/python make_figure.py
```
