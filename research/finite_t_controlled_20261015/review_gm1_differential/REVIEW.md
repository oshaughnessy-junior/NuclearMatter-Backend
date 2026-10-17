# Independent physics review: selected GM1 differential response

**Accept the selected uniform-Hartree quasiparticle prescription and differential mapping. No blocker found within that scope.** This does not close the external matched-GM1 forward gate or qualify integrated rates, a conserving interacting response, or a complete neutrino inventory. No author or upstream file was edited.

## Source and convention audit

I checked the live primary [Guo2020 source, Eqs23–28 and its Eq28 footnote](https://arxiv.org/html/2006.12051v2). Its nucleon dispersion adds the vector potential to the kinetic on-shell energy. Eq26 puts kinetic q* into weak magnetism and pseudoscalar vertices, while Eq27 evaluates their scalar form factors at physical q² and normalizes them with the mean **bare** nucleon mass. Its invariant phase measure uses kinetic nucleon energies, including the correction stated in the Eq28 footnote. The adapter implements this specific prescription. The paper's motivation concerning vector-current conservation is not a license to impose a generic physical-q conserved-tensor reduction; the adapter retains all16 components and makes no such reduction.

For this archived GM1 convention, bare neutron and proton masses both equal939MeV. The weak-current normalization therefore remains939MeV while kinetic spinors use the solved Dirac mass. The physical vacuum neutron-proton mass difference must not be appended to the common-mass EOS. There is no additional Q shift in the adapter. The mapping `nu_i=mu_i−Ui`, `w=omega−Ub+Ua` gives `w+nu_a−nu_b=omega+mu_a−mu_b`, so inverse KMS must use total chemical potentials exactly once. Leptons retain their physical momenta and independent chemical potentials.

EOS momenta and fields use fm units internally; the factory converts fields, mass and chemical potentials to MeV before the response. The tensor integrates MeV momenta directly; an extra fm-to-MeV density multiplier would be wrong. The full baryon spin sum already accounts for spin two, while there is no incident-neutrino average. The independent shell calculation below reproduces the absolute differential prefactor with these conventions.

The nuclear bulk energy is kinetic Dirac energy plus scalar and vector field energies. The sum of single-particle vector contributions is twice the vector field energy. The alternative quasiparticle-energy representation therefore subtracts one vector field energy; adding it again would double count. Electron and muon pair energies are added once. No fluid heating or composition-Q prescription has been implemented or qualified here.

## New held-out forward evidence

`probe.py` solves a new locally stable state `T=8MeV,nB=.22fm^-3,Ye=.17,Ymu=.04`. Its Dirac mass is568.5572415829281MeV; its nuclear root residual is1.80e-15 and nuclear Euler residual is−3.55e-14MeV fm^-3. Thermal antibaryon density divided by baryon density is3.31e-66. Positive-energy response cuts are adequate for this selected state, while the omission of antiparticle/pair cuts remains explicit.

The independent response uses cylindrical transverse momentum and solves the **original energy delta** numerically for longitudinal momentum with Brent's method. It reconstructs vertices directly from the primary-source formula, including distinct physical and kinetic transfers, and uses explicit spinor sums for both lepton and baryon currents. It does not call the adapter's support solver, response integrator, form-factor wrapper, vertex builder or coefficient contraction. It reuses frozen gamma matrices/spinor utilities from the independent free delta oracle; this is not a completely independent software stack or an external GM1 EOS calculation.

| Held-out point | Adapter chi | Independent chi | Adapter j | Independent j |
|---|---:|---:|---:|---:|
|nu_mu, Enu180, El220MeV, cos=.1|62403023.3427846|62403023.342781335|7.808195282915889|7.808195282917670|
|antinu_mu, Enu240, El180MeV, cos=−.2|7218.705521486652|7218.705521484638|9.133852423830847e-13|9.133852423830924e-13|

Units are `s^-1 MeV^-1 per dcos`; these are differential coefficients. Maximum coefficient discrepancy is2.80e-13 relative and full-W discrepancy is2.75e-13 in matrix maximum norm. The directly reconstructed original total-energy shell residual is at most1.12e-10MeV. Both separately integrated occupations satisfy total-chemical KMS; independent radial16→28 refinement passes3e-7 gates. These small discrepancies are selected numerical agreement, not theoretical accuracy.

**35 review checks pass in1.35s.** Additional checks cover the field-energy ledger, nuclear Euler relation, independent lepton neutrality, a91.3MeV common vector/total-chemical shift, correct bare weak mass, and zero-field939MeV free-tensor agreement for both neutrino signs. Changing `(Ye,Ymu)` to `(.19,.02)` at the same Yp leaves nuclear mass and energy unchanged while changing the two lepton chemical potentials separately. The common-vector shift checks microscopic convention invariance only; it does not modify or recompute a Hartree functional.

## Remaining gates

Acceptance concerns a declared Guo mean-field prescription on locally stable uniform branches. It is not a unique conserving interacting theory: there is no RPA, correlation correction, self-consistent weak vertex reconstruction or quantified nuclear/current model uncertainty. Local scalar curvature and the nuclear chemical Hessian do not establish global two-charge phase coexistence. These new held-out shell probes use spacelike kinetic transfer; they do not expand qualification to all timelike/lightlike mean-field points.

The archived rounded-coupling common-mass GM1 EOS still lacks an externally executed, convention-matched finite-T forward reference. Reusing its fields in a second response integral cannot close that gate. Charged-lepton integration/support/tails, crossed decay inventory, volume number/energy moments, coupled material/radiation ledgers, derivatives and production use each require further checks. Preserve the author's qualifications and the external-forward OPEN status when integrating this evidence.
