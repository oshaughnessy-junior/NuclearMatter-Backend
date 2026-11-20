# Independent review: cold GM1 beta composition and Urca triangle anchors

**Accept the selected cold composition/onset physics and restricted pinned-function regression, with an explicit lepton-snapshot newline qualification.** There is no physics blocker. No Urca rate, finite-T width, externally validated GM1 EOS, full-package NMB execution or gradient is qualified by this evidence. All author/upstream files remain untouched.

## Independent forward solve and geometry

`probe.py` uses the shared lepton chemical potential as the independent root variable. It integrates the spin-two lepton momentum density directly, sets `Yp(mu)=[ne(mu)+nmu(mu)]/nB`, and solves `mu−mu_n(nB,Yp(mu))+mu_p(nB,Yp(mu))=0`. This differs from the author's root in Yp. It reuses the frozen internal cold nuclear functional, so it is an independent composition/root/geometry check within the same physical EOS, not an external nuclear calculation.

| Onset | Independent nB [fm^-3] | Independent Yp | Shared mu [MeV] |
|---|---:|---:|---:|
|Muon appearance|.12735715314843768|.04070922145391639|105.6583755000621|
|Electron direct-Urca triangle|.2780845280467474|.13253879089549325|176.8594613188731|
|Muon direct-Urca triangle|.34397893887478986|.16336812331127334|199.49243282133807|

Independent density brackets differ from the author's. The two triangle onset densities agree within3.27e-13fm^-3. Muons are exactly absent below the appearance point and present above it. A finite floating-point root can leave a tiny positive muon density at contact; it is not a resolved physical abundance. Appearance and the two triangle thresholds are distinct conditions.

For each Urca onset I independently reconstructed the proton/lepton surface angle, `cos(theta)=(pFn²−pFp²−pFl²)/(2pFp pFl)`. It reaches1 within1.8e-15 at the root. On the higher-density side, explicit noncollinear proton and lepton vectors sum to the neutron Fermi momentum; below the onset the required cosine exceeds1. Both mismatch slopes are negative: approximately−501.469 and−476.566MeV fm³. The `Yp_crit=1/[1+(1+x_l^(1/3))³]` identity is verified with each actual lepton fraction. The electron-only1/9 value is not substituted for the electron or muon thresholds.

The momenta in these triangles are conserved spatial momenta, not energies or vector-shifted momenta. Nuclear total chemical potentials contain the Hartree fields and common939MeV rest convention; the weak equilibrium equation adds no physical vacuum neutron-proton Q. Equality of electron and muon chemical potentials at cold beta equilibrium does not imply equal number densities. The absent cold muon species is not assigned the finite-T net-zero pair-gas chemical potential0.

## Stability and thermodynamic checks

At four new densities .105,.21,.31,.55fm^-3, central finite differences of the two total nuclear chemical potentials at fixed neutron/proton densities reproduce the author's exact cold Hessian within3.59e-9 relative. Maxwell symmetry and positive local eigenvalues hold. The independent mu-space root derivatives are positive, ranging1.25–3.78, providing a local root/bracket conditioning check. Along the neutral beta branch, independent density differences satisfy `dE/dnB=mu_n` and `dP/dnB=nB dmu_n/dnB`; the latter differs by at most4.17e-10 relative.

The charge-relaxed proton/proton curvature correctly adds the inverse total lepton susceptibility, since electrons and muons share charge through their separate susceptibilities. The scalar-response and vector-isovector normalizations have consistent MeV/fm units. These checks establish local uniform behavior at the selected points, not global coexistence, a crust or a guarantee outside the sampled density band.

## Pinned NMB source audit and qualification

I independently re-fetched [leptons.py](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/25e68182f726e5bd9c0b372c9a667536940fbd07/src/nmbackend/eos/leptons.py), [rmft.py](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/25e68182f726e5bd9c0b372c9a667536940fbd07/src/nmbackend/eos/rmft.py), and [constants.py](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/25e68182f726e5bd9c0b372c9a667536940fbd07/src/nmbackend/constants.py) using the immutable commit. `github_refetch.json` records the returned Git blob SHAs; local byte-derived Git hashes match rmft.py and constants.py exactly.

The lepton file has a real but harmless copy distinction:

| File | Bytes | SHA256 |
|---|---:|---|
|Actual immutable GitHub content|2816|aafaa2bda6af55fd094dd54d2c4cc1b8b14cca45676ac4b730a7d7f79a0960bb|
|Author local snapshot|2817|369daa890f3ddc12587ff89e043ce43e60c0bc232057e87e9ba003bef2e1b0f5|

The local file equals the fetched file **plus exactly one terminal newline**. The original strict byte-identity check failed and is preserved in `initial_strict_byte_identity_failure.json`. The full AST and literal source segments for all three executed lepton functions are independently identical. The actual remote blob SHA is7cd4565470dafb48967546897fc99cc3d8935ad8 and is reproduced from the fetched bytes. This supports the stated unmodified-function-body comparison; do not describe that local lepton file as a byte-exact GitHub archive without the newline qualification. No author file was changed to hide the distinction.

The AST selection executes the actual cold root/function bodies, including NMB's lepton custom root, within a restricted namespace. It does not execute the package import/table/continuation/crust/test-suite surface. The selected pressure function uses the cold beta Euler relation; agreement of that pressure alone is not a second independent validation of the EOS. The pinned tiny-density guards are inactive for the selected states. Its electron mass is the explicitly different legacy .510998950MeV, rather than the newer .51099895069MeV default; muon and common nucleon masses match.

I independently re-solved all five reported pinned densities with the legacy electron mass and compared eight composition/field/thermodynamic values per density, using direct density integration and closed lepton energy expressions. These40 additional comparisons pass. I did not rerun the author's JAX selection, and make no derivative/AD claim. Agreement with one's own pinned NMB conventions remains regression evidence, not external physical validation.

## Warm-state scope and outcome

Direct independent momentum-space FD density integrals recover both baryon and both net pair-lepton densities of all four warm preparations within5.91e-14 relative. The fractions equal their cold onset values. Their nonzero affinities are reproduced: at T8MeV they are(−2.12708,−2.46547)MeV at the electron onset and(−1.57124,−1.77981)MeV at the muon onset for electron/muon sectors respectively. These are fixed cold compositions, not warm beta equilibrium or warm lepton chemical equilibrium. No finite-T rate or opening width follows from the cold triangles or those affinities alone.

**123 selected review checks pass in.32s**, with the earlier strict byte-identity failure retained separately and explained above. Raw evidence, independent fetched content and source hashes are preserved. A preliminary review driver syntax error is also retained and changed no physics or test gate. External matched finite-T GM1 forward values, global phase selection, pairing/correlations, intended warm composition constraints, finite incident-neutrino kinematics and full weak inventories remain separate gates.
