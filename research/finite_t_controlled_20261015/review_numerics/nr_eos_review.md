# Independent numerical review: matched NR FD baryon EOS

15 October 2026. **Accept the EOS and factory at the independently tested modest states**, subject to the declared ideal, nonrelativistic, common-kinetic-mass model. This is a reference for matching occupied baryon thermodynamics to the recoil/blocking kernels; it is not a nuclear interaction model or a domain certificate.

This review executes **46 passing checks**, saved in `nr_eos_review.json`; reproduce with `probe_nr_eos.py`. No author files or dependencies changed. The author's 96-check report was inspected separately.

Two new species points are `(T[MeV],n[fm^-3],offset[MeV])=(5,2e-4,Q),(0.7,8e-4,100)`. Independent scalar momentum quadrature reproduces the density and kinetic energy from the author's energy-space vector integrator. Density inversion, positive fluctuation covariance and tighter quadrature/tail evaluation pass.

Separate canonical-state finite-difference ladders verify Cv and mu_T at fixed density and the coordinate derivatives

```
de/dn|T = mu - T mu_T|n
ds/dn|T = -mu_T|n
dP/dn|T = n/(dn/dmu|T)
dP/dT|n = s+n mu_T|n .
```

These combine the same occupied-dispersion thermodynamics in independent coordinate directions. Constant rest offsets appear in total energy/chemical potential and cancel from Cv, susceptibilities and entropy, as required. The variance expression uses kinetic-energy fluctuation moments and correctly avoids subtracting the common rest offset.

Two new neutral factories at `(T,nn,np)=(5,6e-4,2e-4),(0.7,0.002,0.0005)` independently reconstruct the prescribed neutron/proton densities from the returned transition state's kinetic chemical potentials, offsets and common mass. Net electron density equals proton density; reverse species order and neutrino/antineutrino chemical parameters have the specified signs. Raising both constant baryon offsets by 100 MeV raises baryon energy by `100*nB` and leaves the equilibrium neutrino parameter invariant.

Default offsets `(Q,0)` subtract a constant common baryon rest energy while keeping the selected physical mass splitting. This convention is compatible with a fixed common 939 MeV kinetic mass, but differs from exact relativistic neutron/proton dispersions. A coupled energy equation must retain those same offsets and FD kinetic/pair-lepton energies. Reusing classical `3nT/2` baryon energy where blocking is retained would be inconsistent.

Both species densities must be positive. Vacuum, extreme underflow, interactions, antiparticles, clusters and density-dependent offsets remain outside this slice. Reported T/M and pF/M are useful expansion diagnostics, not error bounds. The factory supplies matched inputs but does not complete crossed channels or integrate a quantum fluid. Collision-energy moments and thermal evolution still require independent conservation and endpoint-integration checks.
