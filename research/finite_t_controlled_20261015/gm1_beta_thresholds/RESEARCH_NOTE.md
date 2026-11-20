# Cold GM1 beta composition and electron/muon direct-Urca Fermi triangles

**Author-frozen onset anchor; independent review required.** This solves one consistent common-bare-mass939MeV GM1 uniform branch with cold electrons and muons. It establishes composition and zero-neutrino-momentum Fermi-surface geometry before any finite-temperature broadening study. It computes no Urca rate, emissivity, broadening width or hydrodynamic evolution.

## Cold composition and conventions

Use the exact internal GM1 cold scalar/number solution at fixed nB,Yp. Its neutron/proton bare mass is common939MeV, with mD=939−Phi, Un=W−R/2 and Up=W+R/2. The nuclear total chemical potentials include rest/field energies. Beta equilibrium without trapped neutrino chemical potentials requires

    mu = mu_n−mu_p = mu_e = mu_mu,
    np = ne(mu)+nmu(mu),  np=Yp nB.

For a cold spin2 lepton of mass ml,

    pFl = sqrt(mu²−ml²), nl=pFl³/[3pi²(hbarc)³],  mu>ml;
    nl=pFl=0,  mu<=ml.

The absent-species branch is an exact zero, not a density floor. Below muon appearance the shared chemical potential lies below m_mu and the muon population vanishes. Its cold chemical potential is not inferred by taking a finite-T net-zero pair-gas root. Lepton energy and pressure are evaluated by direct momentum integrals. The default masses match the existing exact pair-lepton reference: me=.51099895069MeV,mmu=105.6583755MeV.

The beta/neutrality root is independently gated by reconstructed lepton Fermi energies, charge, nuclear scalar-root residual, full cold Euler relation E+P=mu_n nB and local stability. Root brackets restrict the numerical domain and raise if they fail; they do not assign a minimum physical proton fraction. The37-point map covers .08–.80fm⁻³ only. Both nuclear Hessian eigenvalues are positive there; the lowest sampled eigenvalue is38.60045MeV fm³. Canonical scalar curvature is positive. A separately reported charge-relaxed lepton Hessian adds [d(ne+nmu)/dmu]⁻¹ to the proton/proton entry. These are local uniform checks, not a crust, global coexistence or an interacting-model accuracy certificate.

The exact T0 nuclear Hessian uses Fermi-surface susceptibilities and the fixed-density scalar response integral ∫p⁴/[pi²(p²+mD²)^(3/2)]dp. Independent cold-state finite differences at both Urca onsets agree within1.15e−7 relative. Reconstructing baryon momenta from mu_i−Ui and integrating their spin2 momentum densities independently matches the requested neutron/proton densities. No physical vacuum neutron-proton Q is appended to beta equilibrium or to rates; no rate is computed here.

## Three distinct onsets

For neutron-rich homogeneous matter here, pFn is the largest side and each lepton density is no larger than np. Thus the only nontrivial direct-Urca triangle condition for each lepton is

    Delta_l = pFn−pFp−pFl <=0.

The independently bracketed Delta=0 roots use physical conserved spatial momenta, with the emitted neutrino momentum taken to zero. At exact equality the surfaces are marginally collinear. The threshold is an anchor for the leading low-T allowed phase space: strict T=0 rates vanish even on the allowed side. Finite radiation energies, chemical nonequilibrium, thermal tails, pairing and correlations require separate calculations.

| Onset | nB [fm⁻³] | Yp | Ye | Ymu | shared mu [MeV] |
|---|---:|---:|---:|---:|---:|
| Muon appearance |.127357153148|.040709221454|.040709221454|0 at exact contact|105.658375500|
| Electron direct-Urca triangle |.278084528047|.132538790895|.087442430090|.045096360805|176.859461319|
| Muon direct-Urca triangle |.343978938875|.163368123311|.101452497470|.061915625841|199.492432821|

The muon-appearance root is a different chemical threshold from muon direct Urca. Its floating-point root has a tiny positive Ymu~1.6e−21 because mu−m_mu~6e−12MeV; this is reported rather than floored or interpreted as a meaningful resolved abundance. Analytically the population is zero at exact contact.

Electron/muon triangle roots use independent actual sign brackets [.26,.30] and[.32,.38]fm⁻³, respectively, plus decreasing offset sign checks. Residual Delta is below1.59e−10MeV. As an independent geometry identity, with x_l=n_l/np, the threshold proton fraction is

    Yp,crit = 1/{1+[1+x_l^(1/3)]³}.

The roots reproduce it within1.61e−13 absolute fraction. The familiar1/9 is the electron-only x_e=1 limit; it must not be substituted for either npe-mu threshold in this map. The composition and triangle figure is saved as `cold_beta_thresholds.png` and `.pdf` and was visually inspected.

## Actual pinned NMB source comparison

The missing cold lepton module was retrieved through the GitHub connector at immutable NMB commit `25e68182f726e5bd9c0b372c9a667536940fbd07` and is preserved in this leaf's `source_snapshot/leptons.py`. The nuclear functions/constants are the existing full pinned snapshots in `rmf_reference/source_snapshot/`:

- [pinned rmft.py](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/25e68182f726e5bd9c0b372c9a667536940fbd07/src/nmbackend/eos/rmft.py)
- [pinned leptons.py](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/25e68182f726e5bd9c0b372c9a667536940fbd07/src/nmbackend/eos/leptons.py)
- [pinned constants.py](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/25e68182f726e5bd9c0b372c9a667536940fbd07/src/nmbackend/constants.py)

`pinned_compare.py` executes restricted AST selections of the **actual unmodified cold function bodies** in the existing JAX64 runtime, including their scalar solve, beta/charge solve and lepton custom root. It does not retype those formulas or substitute this leaf's root for theirs. Five densities, including both onset neighborhoods, compare Yp,Ye,Ymu,shared mu,Phi,total energy,total pressure and mu_n:40/40 checks pass, maximum relative discrepancy6.64e−13. Actual pinned compositions at the two Urca onsets also reproduce their zero triangles. This is source-convention regression evidence, not independent physical validation or execution of the full NMB package/table/crust/test suite.

The pinned constants use me=.510998950MeV, slightly different from the newer pair-lepton default. The comparison explicitly re-solves this leaf with that legacy electron mass. Default-minus-legacy differences in sampled Yp are below2.83e−15 and in mu below6.83e−13MeV, but they are not silently called identical configurations. The muon and common nucleon masses match. The pinned functions' tiny-density guards are inactive at these selected densities; this leaf introduces no analogous physical density floor.

## Warm preparations and remaining work

At each cold Urca onset the full finite-T GM1/pair-lepton factory prepares T=2 and8MeV states at the **same cold Ye,Ymu**. It gates new field roots, local Hessian/scalar stability, lepton charge roots and Hartree energy functional. These are not re-solved finite-T beta equilibria. Their affinities zeta_l=mu_l+mu_p−mu_n demonstrate that distinction:

| Fixed cold onset | T [MeV] | zeta_e [MeV] | zeta_mu [MeV] |
|---|---:|---:|---:|
| Electron triangle |2|−.12700537|−.14768034|
| Electron triangle |8|−2.12707915|−2.46547376|
| Muon triangle |2|−.09656399|−.10943350|
| Muon triangle |8|−1.57123736|−1.77980604|

Future broadening studies must keep those affinities separate from momentum mismatch and any neutrino energy/distribution. Alternatively they must solve the intended warm beta-equilibrium constraints. No smearing width or warm rate is inferred from these state preparations.

All **327 author checks pass**:281 map/onset/stability/warm checks in2.00s,40 actual pinned-source checks in.72s and6 independent surface-density/source-geometry checks. Charge-fraction residuals are below1.19e−14. Independent momentum lepton densities and full beta-Euler identities are checked across the map. One initial validation driver called abs on an immutable Hessian tuple and stopped while normalizing a comparison; its source and partial report are preserved. Converting that comparison to an array changed no EOS/root formula or tolerance. Initial composition-only and pre-expanded-provenance reports also remain as historical passing records.

Reproduce with `python study.py`, `python check_surface_momenta.py`, `python make_figure.py`, and the explicit existing `/Users/rossma/miniconda3/envs/junior_tools/bin/python pinned_compare.py`. The base numerical runtime and separate JAX runtime are recorded in the manifest. No dependencies were installed and no frozen upstream, production or publication files were edited.

Independent review is required. These onsets belong to a common939MeV GM1 no-sea Hartree ansatz, not the physical unequal-mass free EOS or a modern calibrated uncertainty band. External matched GM1 validation, global phase selection, pairing, warm composition constraints, finite-neutrino-momentum kinematics and actual weak rates remain open gates.
