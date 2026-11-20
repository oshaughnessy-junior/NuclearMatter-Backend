# Spectral free-electron capture emissivity and true neutrino absorption

**Author-frozen prototype, different independent review pending.** Actual separate spectral j(Eν),chi(Eν) are calculated for p+e−↔n+nu_e at the fixed matched T2MeV,nB1e−7fm^-3,Ye.3 free npe bath. Selected detailed differential anchors, incident-energy refinements, isolated tails/azimuth and one-helicity spectral total reconstruction pass their unchanged gates. This is one named reversible CC channel, not a total neutrino opacity, full inventory, new source join or broad state-domain certificate.

## State and physical current

Fresh FD density roots reproduce total mu_n919.8287578100662,mu_p916.8449303084382,mu_e.17446333200826397MeV, with physical mn939.56542052,mp938.27208816,me.51099895069MeV. Neutron/proton particle densities are7e−8/3e−8fm^-3. Actual e−/e+ densities are2.03428636784328e−7/1.73428636784328e−7fm^-3; their net charge is3e−8, not either target population. Full matched-state metadata and equilibrium-antibaryon diagnostics accompany every integrated coefficient. Muons, photons, antinucleon reactions, nuclear mean fields and correlations are absent. The particle-only baryon EOS qualifications remain unchanged.

The full frozen Guo tree current has GF1.1663785e−11MeV^-2,Vud.974,gA1.2723,weak magnetism3.706,vector/axial/pion scales840/1000/139.57MeV and physical-q² form factors. The weak-current mass is the bare average938.91875434MeV. No Dirac/Landau effective mass, measured-lifetime normalization, additional Q shift, Coulomb/radiative correction or surrogate supplies this source. Full source provenance is in SOURCES.md and seven literal transitive physics pins in `spectrum.py`.

## Actual spectral operator and independent occupations

For one active nu_e helicity on a fixed material bath,

```
dfnu(E)/dt = j(E)[1−fnu(E)] − chi(E) fnu(E).
```

j and chi both have units s^-1 per occupation. **chi is true absorption**, whereas j+chi is the fixed-bath relaxation coefficient. In length units divide these time coefficients by c=2.99792458e10cm/s. Neither is itself an energy/volume emissivity.

At incident E, outgoing total electron energy El and cosine c, set omega=E−El and q=|kν−pl|. In the q frame an initial neutron has Ea,pz,pt and the final proton has Eb=Ea+omega and momentum shifted by q. Exact positive-energy support enforces both mass shells, Ea>=mn,Eb>=mp and pt²>=0, including qualified timelike regions. It is not a static target or a smeared threshold.

The source amplitude A_EC is the reciprocal full-current squared amplitude averaged over FOUR initial proton/electron spins. It is evaluated as the frozen invariant polynomial, previously checked against literal spinors and freshly checked here against full tensor contractions. The spin/DOS conversion from unaveraged W gives

```
j_(El,c) = pe/[32 pi³ q E hbar] integral dEa <A_EC>phi
            fp(Eb)[1−fn(Ea)] fe(El)
chi_(El,c) = same prefactor and amplitude integral,
            fn(Ea)[1−fp(Eb)] [1−fe(El)].
```

Both have differential units s^-1MeV^-1 per dcos. Independent positive FD products are integrated in log space; j is never manufactured from chi using KMS. Angular averaging is an actual periodic quadrature of the exact scalar amplitude. The initial spin-two phase measures, four-spin amplitude average and final single-active-helicity normalization are already included; no extra spin factor is added afterward.

Their identity j/chi=exp[−(E−mu_eq)/T], mu_eq=mu_p+mu_e−mu_n=−2.809364169619812MeV, follows from total chemical potentials and exact event energy conservation. It checks signs and occupation bookkeeping, **not** numerical convergence by itself. LTE f=j/(j+chi) is the common-T FD population. At f=0 the source is j; at f=1 it is−chi; the derivative with respect to f is−(j+chi).

## Numerical construction and absolute forward anchors

The fixed-state public `integrate(E)` computes fresh coefficients from the physical chain. `matched()` freshly solves the EOS and verifies exact fixed bath identity; differential calls reject a different bath. Source hashes are checked before physical calls. No saved rate cache supplies a coefficient. Generic physics import aliases and sys.path are restored after importing uniquely named function modules; no frozen driver is executed. This is selected research import hygiene, not a security sandbox.

The outgoing electron endpoint uses El=me+x². Numerical panels include the static central energy only as one of several panel centers, with actual recoil/thermal widths setting further cuts; all samples still use exact moving-target response support. No physical smoothing width or hard stationary-target cutoff is applied. Baryon energy quadrature integrates independent FD logs over exact support; the ordinary scalar amplitude is required positive/finite. The final coefficient preserves finite log versus ordinary underflow/overflow flags. Exact empty support is distinguished from representability loss, without a positive floor.

Four fresh differential points compare the scalar phase implementation against the full16-component complex response: (E,El,c)=(2,3.29,−.41),(5,6.293,.37),(10,11.28,−.23),(2,3.29,.999). The final point probes timelike response support. Three spacelike points additionally compare the independent original delta-shell explicit-spinor oracle. All14 absolute coefficient checks pass, maximum discrepancy5.73e−14 at the unchanged3e−10 gate. These do not calibrate any prefactor.

Three integrated rules (lepton order,angle order,baryon energy order,phi,inner tailT,outer tailT) are(6,8,8,4,40,24),(8,12,12,4,50,30),(12,20,16,8,70,40). All27 selected refinement/KMS/LTE/operator checks pass. Largest coarse→middle coefficient change1.624e−7; largest middle→fine change1.459e−9, versus a3e−6 gate. At E2, isolated inner-tail70→90T, outer-tail40→50T and azimuth8→12 tests pass6 gates, maximum6.22e−15. The latter isolate material/electron tails and angular phase; they are not an analytic bound for all incident energies.

| Eν [MeV] | j [s^-1] | true chi [s^-1] | j+chi [s^-1] |
|---|---:|---:|---:|
|2|.159037562643905|1.76132650852945|1.92036407117335|
|5|.151800496833569|7.53451108217478|7.68631157900834|
|10|.0421695622113088|25.4986495518877|25.5408191140990|

## One-helicity spectral total reconstruction

Use rho_E=E²/[2pi²(hbarc)³] in fm^-3MeV^-1. Empty-radiation number and escaping energy are integral rho_E j dE and integral E rho_E j dE. FD forward/inverse moments use j(1−f) and chi f separately. **No second neutrino DOS or inverse spin multiplicity is inserted.**

The predeclared coarse rule has24 incident Gauss nodes on0..50MeV and the middle material rule; fine has32 nodes on0..60MeV with(12,20,16,4,70,40) material controls. The fine node collection is split into exact disjoint16-node chunks to bound run time, not replace the spectrum with an interpolation/histogram. Each chunk records state, source/reference/review hashes, exact nodes/weights, controls and implementation SHA. Assembly rejects incomplete/duplicated node coverage, mismatched controls/bath/state/source/units and invalid positive/log representations. Partial checkpoint files do not count as accepted chunks. The public spectral API does not use these evidence files.

The coarse collection takes38.71s. Fine takes122.31s across59.20s and63.11s chunks; individual spectral calls stay below5s. All92 reconstruction gates pass: direct KMS at the56 incident nodes, separate forward/inverse number and energy agreement with independently accepted moving-CM event totals, and global refinement. Fine empty totals are

```
R = 3.308860338875078e−7 fm^-3s^-1
Q_nue = 3.160384777809387e−6 MeVfm^-3s^-1.
```

The largest fine-versus-CM discrepancy over empty, LTE and mu_nu=mu_eq−.6T scenarios is1.874e−8 relative; coarse→fine maximum is1.153e−6, both below the unchanged3e−6 gate. Fine number/energy values approach the physical event integral without rescaling. The global refinement changes incident-energy order, upper limit and material rule together; it is not a clean isolation of each individual global error. Independent event-total agreement plus the separate selected-energy controls is the evidence for these selected moments. The physical spectrum above60MeV is **not zero**, and no uniform analytic omitted-tail bound is claimed.

Material dn=+R,dp=−R,dne_net=−R and radiation dn_nue=+R conserve baryon/charge/combined Le. Escaping neutrino energy removes the same rest-inclusive material energy; do not add a second neutron–proton Q. This leaf reconstructs radiation moments and uses the independently reviewed event-total material ledger; it does not independently evolve material thermodynamics or generate all species-energy spectra.

## Frozen evidence and remaining scope

There are204 passing related assertions:14 differential anchors,27 selected spectrum,6 isolated controls,92 total reconstruction,48 admission/operator/representation and17 synthetic assembly checks. These are not204 independent physics measurements. There are no new failed numerical gates in this leaf; the accepted EC total's31 coarse failures and the independent review's coarse-density failure remain untouched.

The initial pilot source/results are preserved when fixed-state admission was strengthened before qualification; differential anchors were rerun with the final source. Assembly control/bath/positive-log validation was tightened before freeze, with old assembly source/results preserved. All17 synthetic attacks/identity checks pass, and physical total results are bitwise unchanged by that metadata hardening. No physical equation, source coefficient or tolerance was changed. QUALIFICATION/MANIFEST/CHECKPOINT pin final bytes and raw node evidence. Source uncertainty is not inferred from numerical method agreement.

Run the owned pure-function drivers in a fresh evidence copy: pilot.py,study.py,controls.py,total.py coarse0,total.py fine0,total.py fine1,total.py combine,isolate_controls.py,check_assembly.py,make_figure.py. Actual CLI arguments are separated, e.g.`total.py fine 0`. Frozen primary collection drivers refuse overwrite. Figures show only detailed finite-quadrature spectra at this state; their connected lines are not an interpolation accuracy certificate. PNG/PDF were visually checked after shortening an overlong units label.

The numerical convenience API admits0<E<=60MeV at this exact fixed state, but qualification is selected differential/integrated evidence, not a uniform numerical guarantee for arbitrarily soft energies or every input. No muon/current-model variations, arbitrary density/T/composition, changed radiation temperature, mean-field/correlated matter, general opacity inventory, beta/PC source join, transparent-equilibrium solution or AD/derivative qualification is granted. A different reviewer must check this new spectral construction and node assembly before broader use.
