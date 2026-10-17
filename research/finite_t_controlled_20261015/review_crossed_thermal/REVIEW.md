# Independent physics review: free relativistic thermal neutron decay/inverse

**Accept the selected differential and stated Enu=.3MeV dilute-npe spectral slice.** No blocker found within that scope. The integrated API guards are restrictions of a numerical rule, not a physical support law or a certificate for every state satisfying them. Total neutrino-energy integrals, time-dilated lifetime and a complete antineutrino inventory remain open. No author/upstream file was edited.

## Current, mass and occupation audit

The actual inverse reaction is `p+e-+antinu_e -> n`; emission is its reciprocal neutron decay. The physical pair momentum is `Q=k+l`, not the capture difference. Ordering the response p→n at positive Q and pairing it with the incoming-electron/antineutrino current is consistent with the crossing and initial electron occupation in [Guo2020, SecIII.2/Eqs36–37](https://arxiv.org/html/2006.12051v2). It must not be replaced by the positron-capture trace. The scalar form factors depend on physical timelike Q²; here physical and kinetic nucleon transfers coincide.

The two actual occupation products are `fp fe (1-fn)` for inverse absorption and `fn(1-fp)(1-fe)` for emission. Final antineutrino blocking is outside j, in `j(1-fnu)-chi*fnu`. Reciprocal WM/PS terms require reversing the transfer as well as taking the Dirac adjoint of the current; the implementation's forward-oriented response and reciprocal amplitude treatment are consistent. The full trace contains the target and electron spin sums, without another electron factor or incoming-neutrino spin average.

Physical mn939.56542052,mp938.27208816,me.51099895069MeV are used throughout; the distinct weak-current mean bare mass is938.91875434MeV. Rest energies are already included in both EOS chemical potentials. Therefore the anti-KMS affinity is `mu_n-mu_p-mu_e`, with no extra manually appended Q. These conventions are not interchangeable with common939MeV GM1 matter. This leaf implements no mean-field crossed response.

For a positive-energy p→n cut the pair satisfies `me²<Q²<=(mn-mp)²`. Thermal motion cannot open a pair violating that invariant bound. Conversely the neutron-rest antineutrino endpoint .7820077025790402MeV is not a global thermal energy cutoff. The exact q=0 response coordinate limit remains explicitly unsupported rather than being declared physically zero.

## New independent forward evidence

`probe.py` independently integrates transverse radius and **both** longitudinal delta roots. It uses original on-shell energies and the original Jacobian `|v_n,z-v_p,z|`. The substitution `r=pstar sin(theta)` resolves coalescence of the two roots. Radial thermal panels differ from the author's energy integral and Q-rest affine-cap rule. Vertices are reconstructed literally, and actual crossed external spinors are evaluated using frozen gamma/spinor utilities; the author's tensor, event amplitude and response integrator are not called by this reference.

The absolute prefactor is independently assembled as `GF²Vud² pe/(256 pi³ Enu hbar)` multiplying the azimuth-averaged two-root radial measure and full spin sum. There is no normalization adjustment or measured-lifetime fit. Explicit decay and inverse amplitudes agree within1.56e-15 relative when their reciprocal currents are evaluated separately.

Fresh neutral npe bath: T=.045MeV,nB=3e-12fm^-3,Ye=.37. Rates below are differential, in `s^-1 MeV^-1 per dcos`.

| Enu,Ee [MeV], cosine | Kernel j | Kernel chi |
|---|---:|---:|
|.27,1.023,.15|.00014177717894798335|1.5674263315992398e-11|
|.43,.863,−.4|.0001447943845966797|5.603903976696851e-10|
|.82,.526,.85|6.741035234593989e-22|1.5147852673759324e-23|

The independent two-root calculation agrees within6.40e-13 relative for both coefficients. Its order12→20 refinement differs by at most5.41e-13. Independent original baryon shells, kernel24→40 refinement and total-chemical anti-KMS pass. The last point is above the rest endpoint and is correctly nonzero; the spectral API still rejects that incident energy because its **numerical rule** has not been extended there. A separate invariant-forbidden pair remains empty despite the positive temperature.

A fresh prescribed mu_e=−1000MeV test, explicitly not a neutral matched bath, yields a finite log chi below ordinary representation. The independent two-root log integral agrees; the kernel reports `value=None,reason=underflow`, not a physical zero or a positive floor.

## Selected spectral evidence and limits

I inspected the retained selected-spectrum ladder and independent event-spectrum report and verified all frozen source/data hashes. The actual main-bath values are j1.6622811423061443e-6 and chi6.405576907923156e-16s^-1 at Enu=.3MeV,T=.03MeV,nB=1e-12fm^-3,Ye=.3. Outer, internal and empirical Doppler-tail refinements are separately reported. The independent event spectrum shares outer panels, so its agreement is not independence of every integration choice; this limitation is accurately stated.

Reconstructing empty-radiation production with the **one-active-helicity** measure `Enu²/[2pi²(hbar c)³]` and dividing by neutron density reproduces the reported per-neutron spectrum. Neither chi nor j+chi can replace j in that production comparison. The quoted near-cold differences from the rest spectrum include genuine Doppler, occupation and finite-T effects, rather than defining numerical error. No total Enu integral was performed by this review or the author. The future unblocked time-dilated total-lifetime target is not an already passed gate, and finite FD blocking would need separate accounting in such a comparison.

**58 review checks pass in1.62s.** The original eight failed uniform-polar-cap comparisons remain preserved; the refined reference resolves a physical thin cap rather than changing a rate or tolerance. The original full-matter muon-susceptibility underflow is also preserved. The explicit npe factory is a declared different species inventory, not a repair/qualification of that full-matter EOS branch.

Acceptance is restricted to the stated main spectral point, its documented near-cold diagnostics, and listed/new differential cases. The allowed integrated T/diluteness bounds do not qualify a generic opacity domain. Endpoint/tail coverage, exact q=0, broader degeneracy/temperature, full Enu number/energy moments, total lifetime, missing positron capture and medium crossed kernels remain open. Coulomb/radiative corrections and many-body physics are absent; no physical uncertainty estimate follows from numerical agreement.

For future coupling, both beta-decay emission and positron-capture emission increase material net electron lepton number by one per emitted antineutrino; their inverses reverse that change. Escaped antineutrino energy is the signed radiation-energy source. Because material EOS energy includes nucleon/lepton rest masses, no separate Q-heating term may be added. Those full-inventory composition/energy ledgers have not been exercised by this leaf.
