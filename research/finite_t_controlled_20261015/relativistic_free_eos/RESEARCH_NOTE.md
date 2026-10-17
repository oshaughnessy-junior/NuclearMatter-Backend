# Free relativistic particle-baryon EOS with independent electron/muon composition

**Author-frozen prototype; independent review required.** `free_baryons.py` supplies a spin2 relativistic ideal **particle-only** neutron/proton EOS and combines it with the frozen relativistic pair-lepton reference. Neutron mass939.56542052MeV and proton mass938.27208816MeV match the existing physical vacuum-channel convention; electron/muon masses are0.51099895069/105.6583755MeV. There are no nuclear interactions, mean fields, nuclei, photons, magnetic fields or baryon-antibaryon contributions to thermodynamics.

## Definition, units and derivative convention

Use natural energy variables in MeV, densities in fm⁻³, energy/pressure/free-energy densities in MeV fm⁻³, and entropy/heat capacity in kB fm⁻³ with kB=1. Total baryon chemical potentials include the physical rest mass: mu=m+nu. For E=sqrt(p²+m²), f=[exp((E−mu)/T)+1]⁻¹ is the occupation **per spin**. The normalization C=1/[pi²(hbarc)³] includes spin2:

    n = C integral p² dp f,
    U = C integral p² dp E f = mn + K,
    P = C/3 integral dp p⁴/E f,
    s = C integral p² dp [−f ln f −(1−f)ln(1−f)],
    F = U−Ts.

All physical rest masses remain in U,F and mu. No common rest subtraction or hand-added Q term is used. The neutron-proton mass difference enters composition energy automatically.

The implementation integrates kinetic energy k=E−m with p²dp=pE dk, explicitly resolving the Fermi surface. Let A_r=C integral p²dp (E−mu)^r f(1−f), r=0,1,2. Analytic responses are

    (dn/dmu)_T=A0/T,
    (dn/dT)_mu=A1/T²,
    (dmu/dT)_n=−A1/(T A0),
    Cv|n=(A2−A1²/A0)/T².

The last expression is a covariance. An unresolved nonpositive susceptibility/covariance raises an error; it is not replaced by a classical heat capacity or a numerical floor. Density inversion brackets the monotone positive **particle** density in nu and gates the forward residual. T must be positive, and exactly zero baryon species is explicitly rejected. No T=0 branch is claimed.

## Independent lepton fractions and EOS closure

`matter_state(T,nB,Ye,Ymu)` uses Yp=Ye+Ymu and nn=(1−Yp)nB,np=Yp nB. The declared nB is the sum of particle baryon densities in this particle-only ansatz. It is not an implemented particle-minus-antiparticle baryon EOS. Require0<Yp<1; pure neutron/proton endpoints are rejected without floors. Net Ye or Ymu may individually be negative if their sum obeys this condition. Charged leptons retain their exact existing pair equilibria, including negative net charge and a zero-net pair plasma. Ye is never silently equated to Yp.

The returned immutable `Matter` contains all four species, total U,P,s,F and fixed-composition Cv. Lepton chemical potentials are total with explicit masses. `zeta_e=mu_e+mu_p−mu_n` and `zeta_mu=mu_mu+mu_p−mu_n` are separate equilibrium radiation affinities, not enforced constraints. No beta equilibrium is imposed.

`matter_derivatives(m)` returns u=U/nB and derivatives with the other state inputs held fixed. With A_i=mu_i−T(mu_i,T)|ni, baryon n_i is particle density and lepton n_i is net density:

    u_T=Cv/nB,
    u_Ye=Ap−An+Ae,
    u_Ymu=Ap−An+Amu,
    u_nB=[(1−Yp)An+Yp Ap+Ye Ae+Ymu Amu−u]/nB.

`temperature_from_energy(U,nB,Ye,Ymu,Tguess)` inverts total physical energy at fixed composition. Its bracket must remain inside the represented, antiparticle-qualified ansatz; a bracket leaving that domain can reject even when some more carefully chosen bracket would contain a valid root. It is a tested local closure, not a global state-inversion certificate.

## Antinucleon diagnostic and exclusion

For every baryon state, independently integrate nbar_equil(T,−mu,m) as a **diagnostic**, without adding it to thermodynamics. The default contract rejects nbar_equil/n>1e−8. This is a declared ansatz gate, not a proof of equilibration or a physical density floor. Current selected states are much safer than that limit.

The diagnostic retains finite `log_antiparticle_density` and log ratio even when the ordinary density underflows. `antiparticle_density_representable=False` distinguishes numerical zero from physical absence. A scaled occupation integral pulls out f(E=m), evaluating its ratio analytically on the dilute branch; it avoids subtracting two log occupations near−1.9million at the coldest test. Independent scaled-Bessel Maxwell-Boltzmann log densities check that extreme-suppression limit. A user-modified antiparticle threshold is a different declared qualification choice; only the default is tested here.

## Selected evidence and failures

All **143 final author checks pass**:134 main checks in11.45s and9 extra checks in.19s. The source is self-contained NumPy/SciPy with the existing pair-lepton module; no dependencies were installed.

Four particle species states span(TMeV,nfm⁻³)=(5,1e−8),(2,.002),(.05,.08),(40,.01), with physical neutron/proton masses. Independent momentum-coordinate quadratures reproduce the energy-coordinate n,K,P,s,Cv to1.13e−13 relative, after gating the unperturbed density root. Separate tail/accuracy refinements change selected outputs by at most1.78e−14. Euler U+P−mu n=Ts agrees to2.81e−10 relative; its cold subtraction of large rest terms is explicitly less well conditioned than K or Cv.

Decreasing finite-difference steps check Cv, susceptibility and mu_T; largest errors are5.52e−7,2.70e−9 and9.81e−7 relative. The chosen cold steps are larger than the discarded underresolved steps, while their tolerance is unchanged. Matter u_T,u_nB,u_Ye,u_Ymu checks at four independent compositions agree within7.84e−8 relative. Energy-to-temperature recovery agrees within1.76e−11 relative. The tests include Ymu=0, Ye=0 and negative Ymu with positive total Yp, preserving their pair gases.

Analytic witnesses include relativistic dilute Bessel n and mean energy (finite-FD corrections~5.26e−9), cold relativistic E0,P0 and Sommerfeld Cv, and a decreasing-temperature T² thermal-energy ladder. The cold Cv differs from its leading Sommerfeld expression by at most5.51e−6 at the selected temperatures. Matched physical-mass NR comparisons approach the NR gas to2.67e−5 at the tested small T/m and momenta; that difference is a physical approximation correction, not quadrature error. These are selected limit witnesses, not uniform domain bounds.

Example T=2MeV,nB=.001fm⁻³,Ye=.15,Ymu=.03:

| Quantity | Value |
|---|---:|
| neutron total mu [MeV] |938.9902236423|
| proton total mu [MeV] |934.3379313985|
| electron total mu [MeV] |32.0340826887|
| muon total mu [MeV] |104.9278416504|
| total U [MeV fm⁻³] |.949625798597|
| total P [MeV fm⁻³] |.00349394072051|
| total s [kB fm⁻³] |.00350699031312|

Preserved initial failures remain explicit:

- A T37MeV,mu=mp−18T dilute-Bessel candidate has too many equilibrium antiparticles for the default1e−8 gate and is excluded. The in-domain T25 candidate uses the same dilute fugacity; the original exclusion has a regression.
- The original cold diagnostic lost digits by subtracting log occupations near−1.9million, causing the quadrature error gate to reject. The scaled-occupation algebra was repaired without floors or weaker error criteria. Original source, partial results and failure record are retained.
- Two very small cold mu_T finite-difference steps lose digits in total-mu subtraction and fail their unchanged tolerance. Their raw failures are preserved; larger decreasing steps resolve the analytic derivative.
- A T.1MeV energy differs from the exact cold energy by7.24e−7 due to genuine thermal energy and fails a3e−7 near-zero approximation gate. It is not relabeled an EOS defect or hidden by loosening that tolerance. The colder.05,.025,.0125MeV sequence passes and has the expected T² scaling.

## Tensor integration contract and reproduction

For the proposed free response, use each species' `.temperature`, `.mass`, `.chemical_potential` and particle `.density`. The quasiparticle vector potential is U_i=0; spinor mass is that physical species mass. `.kinetic_chemical_potential=mu−m` is rest-subtracted and differs from a Hartree kinetic chemical potential that still includes mD. Do not pass the rest-subtracted field to an occupation expecting total mu. Weak-current Mweak remains a separate bare current convention; this EOS does not select or modify it.

Run `python validate.py` and `python extra_checks.py` from this directory or via absolute paths. Both scripts write raw results. `MANIFEST.json` pins code, frozen lepton source and the physical-mass source convention. No response tensor, finite-T weak rate, field model, external interacting EOS calibration, hydro or production release is implied by this EOS qualification.
