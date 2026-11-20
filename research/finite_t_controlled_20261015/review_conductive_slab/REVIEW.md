# Independent review: controlled conductive electron thermal slab

**ACCEPT the selected fixed-background, local-FD, electron-only thermal diffusion prototype and its reported finite-grid ledgers.** There is no selected-scope blocking finding. The nonlinear transient continuum/spatial accuracy gate remains OPEN. This review does not qualify a microscopic local-energy-thermalization mechanism, realistic crust/core conductivity, arbitrary material states, or a full hydro/transport backend.

Author source `conductive_thermal_slab/slab.py` SHA256 is `036e45427ba33eef06cd94127487d7afac33220e50e87e6d8bf5448a40965317`. All frozen author files and upstream source hashes were checked without edits. **55 review checks pass**:17 new diffusion/steady/nonlinear checks and38 source/unit/thermodynamic/ledger/figure checks. The count includes source pins and algebraic controls, not55 independent physical benchmarks. `probe.py`, `audit.py`, `results.json`, `audit.json` preserve executable probes and raw evidence.

## Physical closure and material conventions

The conductivity is the actual screened static-ion Lorentz-gas open-circuit covariance `(K2−K1²/K0)/T`. It is not the zero-electrochemical-field coefficientK2/T. With zero particle/charge current, the heat flux equals the total material energy flux because the chemical energy current is zero; no missing Peltier contribution or independent rest-mass source is needed. The electrochemical field is a slaved quasineutral closure with no independent field-energy evolution. Electrical work is j·E=0 in this ansatz.

Electron energy, entropy and Cv come from the same spin2 particle-only relativistic FD EOS as the transport moments. Density is fixed, and the rest-energy density is a constant that cancels from thermal differences. New fixed-ne derivative probes atT/T0=.96 and1.037 reproduce Cv and du=Tds, using actual EOS energy and entropy. The positive open-circuit covariance is verified at both states. Their pair diagnostics remain below the explicit electron-only exclusion gate. Neutrality fixes external ion densityni=ne/Z; no ions move, store heat, perform mechanical work or screen through an independent response. Adding a3ni/2 heat capacity would change this named model and its diffusion time.

Elastic static-ion scattering relaxes angular distortions while conserving each electron energy. It cannot establish a FD shape from a general isotropic distribution. Local FD is an additional explicit assumption in the author note and figure. A new energy-redistributing collision mechanism could also change conductivity and needs its own controlled ordering. Small angular-scattering Knudsen number is not evidence for the missing energy-thermalization rate.

## Independent units and kinetic scales

The dimensionless temperature isT/T0; W=[u(T)−u(T0)]/(Cv0 T0), with dW/dtheta=Cv(T)/Cv0. The conserved equation is dW/dtau=−div(q), q=−k grad(theta). The normalization yields tau=t*kappa0/(Cv0 L²) when all quantities first use consistent natural units.

I independently reconstructed SI quantities through conductivity inW/m/K and heat capacity inJ/m³/K, rather than repeating the author's natural-time expression:

    Cv_SI = Cv_fm^-3 * 10^45 * kB_J/K,
    kappa_SI = kappa_MeV² * kB_J/K / [hbar_MeV*s * (hbarc)_MeV*m].

The results are Cv=1.320316653891225e14J/m³/K, kappa=1.1401091720383043e13W/m/K, D=.08635119224454112m²/s, andt0=L²/D=1.158061601706741e−7s for the chosen100µm length. These reproduce the author's scales without an omitted kB orfm↔m factor. Fermi velocity2.926149284440446e8m/s times collision time3.0096235021827795e−18s independently reproduces the mean free path and Kn. The tiny2.23e−10 hbar*c versus namedhbarc mismatch is ordinary rounded-constant consistency, not a physical discrepancy.

These are selected-state Fermi-surface scales. They do not certify every thermal electron's mean free path or the physical validity of Born scattering/static ions. The numerical agreement is within a shared conductivity/EOS model; it is not external calibration or a physical-uncertainty estimate.

## New analytic and steady witnesses

For13cell centers and two held-out sine modesm=1,3, an independently assembled diffusion matrix has endpoint diagonal−3/dx² and interior diagonal−2/dx². Its matrix exponential at tau=.017 agrees with sin(mπx)exp[−4sin²(mπ/(2N))*tau/dx²] to6.67e−16 and2.23e−16 absolute. This checks the half-cell reservoir geometry and discrete decay using a different integrator. The author's constant-coefficient spatial/time tests remain separate from the nonlinear physical problem.

A new actual-conductivity steady reference uses reservoir temperatures1.06/.95. Independent adaptive quadrature of G(theta)=integral_.95^theta k(z)dz and bracketed roots atx=.07,.31,.58,.91 produce a constant flux. Direct finite-difference gradients of the resulting temperature profile, multiplied by actual local conductivity, agree with the integrated flux to7.25e−11 relative. The quantity integrated is the actual conductivity, not a fitted diffusivity or frozen constant-k surrogate.

## Independent nonlinear energy and entropy

The fresh five-cell nonlinear case starts fromtheta=1+.025sin(2πx)+.015cos(πx), with fixed reservoirs1.06/.95, and evolves to tau=.006. It uses an independently authored second-order SSPRK2 update in conserved W and bracketed EOS roots. Each face heat flux uses a three-point quadrature of the actual conductivity integral between neighboring temperatures divided by their distance. This differs from the author's explicit-Euler/harmonic-face scheme. No conductivity or EOS interpolation is used.

Runs with12 and24steps both remain within temperature bounds, have positive incoming and outgoing reservoir-energy integrals, conserve energy when reconstructed from inverted EOS temperatures, and produce positive material-plus-reservoir entropy at every saved step. The maximum normalized energy residual is at most1e−12. The final temperature difference is1.493e−6 between the two time rules. These are finite-grid/time checks, not a continuum error bound or a comparison of two methods' exact spatial discretizations.

The entropy was independently reconstructed from s(T), not obtained by assigning it the face production. For the12step run, actual total entropy production is1.1849736412831372e−4, whereas the independently time-integrated positive face sum is1.1849803525833514e−4. The discrepancy is−6.7113e−10; its magnitude decreases by a factor.1444 at24steps. Thus energy telescoping is exact up to EOS inversion while finite-step entropy balance is demonstrably not an identity imposed by the update. Two time resolutions support decreasing residual; they do not establish an asymptotic order theorem.

The signs are physical: qL is positive into the material from the left, qR is positive out on the right, and entropy input is qL/thetaL−qR/thetaR. The reservoirs' entropy change has the opposite sign. The total entropy production is material entropy change minus this boundary input. The face expression sum[q Δ(1/theta)] is nonnegative with heat flowing down temperature, including the boundary half cells.

## Original endpoint and figure audit

I separately reconstructed the author's64step endpoints from saved temperatures and the actual EOS. The isolated material entropy increase is.0011661374132883946 with zero boundary flux. The reservoir material entropy change is.0004454311722597113, boundary entropy input−.0009708736378574189, and total entropy production.0014163048101171302. Omitting the reservoir contribution would alter the physical interpretation. Reconstructed energy also closes against separately accumulated incoming/outgoing energy.

The original finite-time entropy-versus-face discrepancies are−1.8294e−6 and−1.33893e−5; they are nonzero and properly retained as numerical temporal discrepancies. Neither their existence nor energy conservation establishes nonlinear spatial convergence. Likewise conservation of energy does not imply conservation of mean temperature under nonlinear u(T).

`thermal_profiles.png` was visually inspected. It clearly labels eight calculated cells, local FD imposed, electron Cv only, initial versus tau=.02 profiles, and a separate independent steady curve. Its provenance points to the exact frozen final raw-results hash. The transient curve is not labeled steady and joined cell values are not presented as a continuum accuracy certificate. Figure axes, units and temperature labels are consistent with the data and note.

## Retained failures and remaining gates

The six original entropy-assertion failures are preserved. They compared a positive entropy quantity to zero with an absolute-error test; the inequality's physical sign was already positive. The source/cache/CFL changes before author freeze are separately documented; this review qualifies the current pinned source, not every prior version.

The source's interiormax(k)/min(Cv) CFL diagnostic and selected maximum-principle checks are adequate evidence for the demonstrated mild-temperature-contrast runs. They are not a universal large-contrast boundary stability theorem: reservoir face conductivities and heat capacities over the full inversion interval need to enter any such broader proof or guard. This is an extension advisory, not a selected-state failure.

Next gates remain: nonlinear dynamic spatial refinement with controlled time error; a physical FD-energy-thermalization mechanism and its effect onkappa; broader temperature/density/gradient transport qualification; correlated/mobile-ion, electron-electron or realistic crust/core physics as separately declared models. No production NMB conductivity is replaced, and no hydro, photons, neutrinos, ion thermal evolution or coupled field dynamics are qualified.
