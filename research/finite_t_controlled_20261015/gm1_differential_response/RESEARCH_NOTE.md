# Selected GM1 Hartree differential charged-current response

**Author-frozen prototype; independent review required.** This adapter connects the existing exact internal finite-T GM1 nuclear EOS to selected all-component differential charged-current responses under the declared Guo mean-field current prescription. It is a phenomenological uniform-Hartree quasiparticle ansatz, not external physical validation of GM1, a conserving interacting response, or a charged-lepton integrated opacity/source. No production code, paper or frozen upstream source was changed.

## State, thermodynamics and forward gates

`matter_state(T,nB,Ye,Ymu)` solves the frozen GM1 branch at independent Yp=Ye+Ymu. Neutron/proton bare masses are the archived **common939MeV** convention. It does not add the physical vacuum neutron-proton Q. Electron and muon pair gases have their own explicit physical masses and independent net fractions; Ye is never silently identified with Yp. No weak equilibrium is imposed.

The factory re-evaluates the nuclear scalar/density root residual, exact fixed-T chemical Hessian, canonical scalar curvature, field-energy decomposition, nuclear Euler identity, antiparticle diagnostic and lepton neutrality before a selected response is called. The two nuclear states are:

| T [MeV] | nB [fm⁻³] | Ye | Ymu | mD [MeV] | lowest nuclear Hessian eigenvalue [MeV fm³] |
|---:|---:|---:|---:|---:|---:|
|5|.20|.18|.02|594.025075438|813.596030538|
|12|.24|.20|.05|544.111077149|885.130932308|

Both canonical scalar curvatures are positive (.13231925 and.14100454fm⁻²). Maximum normalized forward residual is3.34e−15 and nuclear Euler residual1.21e−13MeV fm⁻³. These are robust **local nuclear** stability checks, not global two-charge phase coexistence or a uniform parameter-domain certificate. The still-open external matched GM1 forward gate remains open.

Bulk nuclear energy comes directly from the Hartree functional

    U_nuclear = K_Dirac + U_scalar + U_vector.

The quasiparticle vector-potential sum Un nn+Up np equals2U_vector. Thus summing kinetic-plus-vector particle energies and then adding the unchanged complete field energy would double count. The equivalent correct form is U_quasiparticle+U_scalar−U_vector. These identities are checked explicitly. Material energy/pressure/entropy add the two pair-lepton gases once. No additional manually inserted potential/Q heat term is supplied. No fluid update is performed.

The frozen GM1 EOS contains thermal baryon antiparticles, whereas this response uses positive-energy particle-hole cuts only. Selected antiparticle/net-baryon ratios are8.05e−109 and3.14e−43, and the factory rejects ratios above1e−8. This diagnoses the selected omission; it does not implement pair-creation cuts.

## Explicit physical/kinetic current mapping

For an ordered a→b absorption pair,

    Ei_total = Ei_star + Ui,
    Ei_star = sqrt(p²+mD²),
    Un=W−R/2, Up=W+R/2,
    nu_i = mu_i_total−Ui,
    w = omega_physical−(Ub−Ua).

Here nu_i still includes mD, unlike the rest-subtracted chemical potential in the older NR reference. All fields/masses/chemical potentials passed to the response are MeV; only EOS densities are fm⁻³. The tensor integrates natural momentum units directly and has MeV² units. No extra hbarc density conversion is multiplied into it.

The adapter reuses the independently reviewed frozen free tensor's support, energy/azimuth integration, full trace and scaled-log representation. It supplies kinetic masses mD, kinetic chemical potentials nu_i and kinetic transfer q*=(w,qvec). A small **explicit form-factor wrapper** evaluates GV,GA,F2,GP at physical q²=omega²−qvec² while the vertex receives q*. Its ignored kinetic argument is intentional: physical and kinetic invariants are held separately for each differential point. The WM and PS vertices are

    iF2(q²) sigma_munu qstar_nu/(2Mweak),
    −GP(q²) gamma5 qstar_mu/Mweak,

with **Mweak=939MeV**, independent of mD and the density-defined Landau masses. Physical and kinetic transfers coincide only in the matched zero-field limit. This is the source-specific [Guo prescription, Eqs23–28](https://arxiv.org/html/2006.12051v2), not a derivation that uniquely resolves in-medium vertex physics.

The full unaveraged target-spin trace and all16 complex W components are retained. No generic charged-current conservation relation removes components. The charged-lepton tensor and four-momenta use the **physical** neutrino/lepton energies and the physical spatial q; vector nucleon potentials are not inserted into lepton kinematics. The current is free-quasiparticle Hartree only: no RPA, correlations or vertex-conservation reconstruction is claimed.

Absorption and inverse-emission occupations are integrated separately. Their pointwise logarithmic ratio is

    −[w+nu_a−nu_b]/T = −[omega+mu_a_total−mu_b_total]/T.

Consequently j/chi=exp{−[Enu−(mu_l+mu_b_total−mu_a_total)]/T}. This uses total chemical potentials with no second potential shift. For antineutrino absorption p→n, the charged positive lepton uses −mu_l and the antineutrino trace. The output remains differential, in s⁻¹MeV⁻¹ per unit cosine. It is neither a true-opacity integral nor relaxation opacity.

## Selected differential results and qualification

At the first state use electron and muon neutrinos; at the second use an antimuon neutrino. El is total charged-lepton energy.

| Flavor | Enu,El [MeV] | cosine | j | chi |
|---|---|---:|---:|---:|
|νe|110,160|.3|1042.92674302|11464.4335424|
|νmu|160,210|−.25|.00151877881853|46351092.0365|
|antiνmu|200,148|.2|.0000159179456430|81225.3697405|

The n→p kinetic transfer is w=2.21271901384MeV although physical omega=−50MeV. The p→n transfer is w=−.21271901384MeV although physical omega=52MeV. These examples make the convention distinction consequential. The electron can gain energy from the isovector potential difference without adding a physical vacuum Q.

All **217 author checks pass**:97 main checks in2.61s,84 mapping/original-shell checks in.002s and36 separate-control refinements in.70s. They include full W Hermiticity/positive semidefiniteness, separate j/chi KMS (max4.00e−14 absolute log error), physical lepton shells, original total baryon energy deltas (max1.14e−13MeV), physical form-factor values and kinetic support. Refinement24/8/50T→40/16/80T and separate energy-only, azimuth-only and tail-only changes pass unchanged3e−7 gates. The combined-refinement differences are~1.8e−15 at these smooth selected points; this is not a broad error certificate.

Adding37MeV to both vector potentials and both total chemical potentials gives exactly unchanged tensors and coefficients at the sampled points. This is a microscopic common-energy-shift test, not a recomputed shifted Hartree energy functional. Zero potentials with mD=Mweak939 and the same total chemical potentials match the frozen **common-mass** free tensor for both neutrino signs, with no physical-Q comparison or fitted scale.

Deliberately replacing Mweak by mD changes chi by18.17%,28.49%,12.18% for the three points. Deliberately evaluating the form factors at qstar² instead of physical q² changes W by~1.00%,1.24%,1.57%. These compare **different prescriptions** and verify that the adapter does not silently conflate them. They are not estimates of nuclear/current model uncertainty, and the alternate results are not promoted as corrected physics.

One initial mapping-check report failed because a real lepton four-vector JSON list was read as a complex real/imag object. The initial driver/error is preserved, the loader was corrected and the same data replayed. No EOS, kernel, physical result or acceptance criterion changed.

## Interface and next gates

`Matter.ordered(antineutrino=False)` returns immutable `OrderedState(T,mD,Ua,Ub,mu_a_total,mu_b_total,weak_bare_mass)`. The differential call takes physical Enu,El,cosine and explicit charged-lepton mass/chemical potential, making flavor/charge conventions visible. `response` returns the inherited all-component/scaled-log tensor plus complete physical/kinetic mapping metadata. `Matter` exposes the original Hartree energy functional, not a quasiparticle sum.

Reproduce with `python validate.py`, `python check_mapping.py`, and `python refine_controls.py` from this directory or via absolute paths. Raw tensors, exact state roots, forward/stability gates, alternate-prescription diagnostics, controls and source hashes are retained. The underlying GM1 source convention remains pinned to archived NMB commit25e68182f726e5bd9c0b372c9a667536940fbd07 through the existing nuclear reference.

Independent review must check this new physical-q²/kinetic-q* adapter separately, preferably with independently reconstructed vertices and original delta-shell phase space. The already accepted free tensor does not independently validate a new in-medium prescription. External GM1 validation, global coexistence, complete crossed inventory, charged-lepton integrated rates/tails, material/radiation source ledgers, gradients, interacting response uncertainties and production use remain separate gates.
