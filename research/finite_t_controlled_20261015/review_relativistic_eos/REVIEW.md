# Independent numerical review: free relativistic baryon + pair-lepton EOS

**Accept the selected free-gas particle-only scope. 59/59 new checks pass.** This review changes no author code and does not replay the author's143 checks. `probe.py` supplies50 checks in4.49s; `check_signed.py` supplies9 additional cheap checks. Source and preserved-failure hashes are verified unchanged. The independent review's initial JSON-array reporting failure is retained with its original driver; the corrected serializer changes no numerical gate.

## Forward values before derivatives

I independently integrated the original **momentum variable**, including spin2/[pi²(hbarc)³], rather than the author's kinetic-energy variable. Separate occupation factors, entropy, energy, pressure and susceptibility/thermal moments were implemented in this review. Generic Brent roots of these momentum integrals independently reproduce two new physical baryon states:

| Species | T [MeV] | n [fm^-3] | author total mu [MeV] | Cv [kB fm^-3] |
|---|---:|---:|---:|---:|
| neutron |3.7|.0034|940.8309934223895|.004728475415830258|
| proton |.19|.021|953.2663313433724|.0013227687665756233|

Unperturbed forward densities are gated before all derivative comparisons. The maximum independent forward density discrepancy is2.85e−12 relative; independently solved chemical potentials differ by at most1.70e−11MeV across species and the assembled state. Energy, pressure, entropy, susceptibility, fixed-density Cv and fixed-mu n_T agree within their unchanged3e−9 gates; independent Cv discrepancies are at most1.21e−13 relative. The covariance Cv formula has the correct units and fixed-density subtraction: Cv=A2/T²−A1²/(A0 T²), with positive susceptibility. No classical replacement is used.

## Independent composition, derivatives and closure

A new assembled state uses T3.3MeV,nB.0023fm^-3,Ye.118,Ymu.047, with Yp=.165. Independent momentum-density roots are solved separately for n,p,e±,mu±, then independently summed U,P,s,Cv agree with author values within5e−9. Total U=2.193576392917779MeVfm^-3 includes physical neutron/proton rest masses. Feeding that independently computed energy to the author's local temperature inversion recovers3.299999999991593MeV (2.55e−12 relative error).

Two decreasing finite-difference steps independently re-solve the assembled EOS in T,nB,Ye,Ymu before comparing analytic specific-energy derivatives. Errors shrink by approximately4, as expected for central differences. At the finer step the relative errors are2.99e−8,1.08e−7,2.63e−6,7.66e−8 respectively; all pass the predeclared2e−5 gate. The composition terms correctly change proton fraction through Ye+Ymu while using separate lepton energy responses. Physical masses remain in U and total mu; a separate manually added vacuum Q would double count the neutron/proton composition rest energy.

Additional new signed compositions are(T4.1,nB.0011,Ye−.027,Ymu.19) and(T2.9,nB.0017,Ye.22,Ymu0). Independent scalar momentum integrations confirm both signed net lepton densities and charge closure np=ne,net+nmu,net. The zero-net muon species retains its pair gas, with mu_mu=0. The two zeta values differ by mu_e−mu_mu; neither enforces trapped equilibrium or transparent kinetic source balance. The density here is explicitly particle baryon density, not net baryon density with an antibaryon gas.

## Antiparticles, boundaries and preserved failures

A held-out cold state T=.0013MeV,n=2e−8fm^-3 has a finite antiparticle log density agreeing with an independent scaled-Bessel Maxwell-Boltzmann expression within3e−9 in the log. Its ordinary density is unrepresented zero, with the representation flag explicitly false. This is a diagnostic for an omitted species, not an included contribution to thermodynamics. The default nbar/n<=1e−8 ansatz rejection is active; the original T37MeV,mu=mp−18T case is independently confirmed excluded. The reviewer does not change that threshold or reinterpret the original failure as a supported high-temperature state.

Zero particle density and either zero-baryon-species composition endpoint are correctly rejected. NaN density is rejected. The public code rejects unresolved positive covariance/susceptibility rather than inserting floors. Temperature inversion is local: its automatic bracket can encounter the antiparticle ansatz exclusion even if a narrower valid bracket exists, as the author note explicitly states. This is an accepted limitation, not a global inversion guarantee.

The author preserved its initial cold scaled-log quadrature failure, two underresolved total-mu finite-difference failures, and the genuine finite-T correction that failed an overly cold T=.1 approximation test. Those original files and hashes remain unchanged. The later passing cold ladder is evidence for the stated selected asymptotic limit, not grounds to erase the original failed gates.

## Qualification boundary

Accepted: these explicit relativistic free particle-baryon and independent pair-lepton states, their local thermodynamic responses and local fixed-composition energy inversion, plus the existing author's separately disclosed selected limits. This does not establish a uniform density/temperature/composition error rectangle. Nuclear interactions, Hartree fields, baryon-antibaryon thermodynamics, photons, zero-temperature/zero-baryon branches, reaction rates, tensor normalization, external interacting EOS agreement, evolution and hydro remain separate gates. Coupling to the free tensor must pass **total** species chemical potentials and physical species masses; the rest-subtracted kinetic chemical potential is not interchangeable with total mu.
