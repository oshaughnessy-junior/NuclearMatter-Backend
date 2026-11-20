# Independent review: finite-T muon research contract

**Accepted as a research/implementation contract.** No new weak kernel, muon opacity, external-code parity or muonic astrophysical result is qualified. The proposed variables, signs, reaction crossings, thermodynamic closure and vacuum-support gates are consistent in the stated nucleonic, pair-equilibrium, no-oscillation ansatz. The author's frozen files were read-only and their hashes remained unchanged.

`probe.py` supplies38 passing checks in9.00s. At two newly assembled states, independentYe/Ymu enter the proton density only throughYp=Ye+Ymu; the separate lepton charge roots reproduce that proton density. Central finite differences of the assembled total energy inT,Ye,Ymu agree with the proposedCv/nB,Ap−An+Ae andAp−An+Amu derivatives to1.43e-8 relative. These checks reuse the already reviewed individual ideal-gas solvers; they establish the new composition assembly and calculus, not new external EOS validation. Independent prescribed lepton fractions do not require equal chemical potentials.

All16 displayed semileptonic/leptonic reactions conserve baryon number, electric charge and both electron/muon family lepton numbers. In particular mu− decay increases materialYe and decreasesYmu while emitting antinue andnumu; charge andYp stay fixed. The two trapped beta affinities also satisfy the decay equilibrium condition identically. Transparent empty-radiation balance is a kinetic source condition and cannot be inferred from these chemical identities. Mixed-lepton inverse kernels depend on partner neutrino occupations, as specified.

## Primary-source checks

The named mixed-flavor crossings agree with [Sugiura et al., Table1 and §2.1](https://academic.oup.com/ptep/article/2022/11/113E01/6680255). That source also retains partner occupation factors and distinguishes pair/decay from scattering kernels. The contract's species-resolved event ledger is appropriate; a flavor-blind scalar cooling term would lose its conserved family numbers.

The distinction between full massive-lepton phase space and current corrections is necessary. [Guo et al., §III–IV](https://arxiv.org/html/2006.12051v2) treats muon kinematics, weak magnetism, pseudoscalar and form factors together; replacing an electron mass alone cannot reproduce that reference. [Roberts/Reddy, §III.2](https://arxiv.org/html/1612.02764v2) explicitly cautions against imposing the conserved-vector-current reduction with unequal neutron/proton masses and self-energies. No correction factor is inferred here from their selected numerical states.

The proposed bare decay normalization and massless-electron spectral normalization follow [PDG2025 Eq57.3](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-muon-decay-params.pdf). They must remain separated from measured lifetime/radiative corrections. This review does not implement or calibrate a decay matrix element.

The application references are used with suitable limits. [Gieg et al.2024](https://arxiv.org/html/2409.04420v1) labels its charged-current treatment elastic; [Ng et al.2024](https://arxiv.org/html/2411.19178v2) separates muon neutrino species and uses full kinematics; [Gieg et al.2026](https://arxiv.org/html/2604.14225v1) adds inverse muon decay with declared pair-neutrino closure. Their differing approximations do not establish a universal cooling correction. [Weakhub](https://arxiv.org/html/2309.03526v2) was also accessed. The [Guo code landing page](https://git.gsi.de/nucastro_archive/supernovamuonicrates) is accessible, but neither its implementation nor any matched run was retrieved for this review.

## Independent vacuum support

Rather than merely re-evaluating the printed threshold formula, the probe minimizes the final two-mass lab energy at fixed total incident momentum and solves its equality with the initial rest-target-plus-neutrino energy. It recovers:

| Process | Free rest-target threshold [MeV] | Static threshold [MeV] |
|---|---:|---:|
| numu+n→mu−+p |110.1613730132|104.36504314|
| antinumu+p→mu++n |113.0473105953|106.95170786|

Both agree with the invariant formula to better than1.4e-15 relative. Vacuum neutron muon beta decay has negative mass-energy excess, so it is forbidden. These physical unequal-mass relativistic vacuum checks must not be presented as a normalization benchmark for the common939MeV quadratic-dispersion response without explicitly taking the relevant limit.

For inverse muon decay on an electron at rest, directly solving lab energy closure gives10923.1449285MeV. This is **10.923GeV**, not a generic dense-electron threshold. For illustration, a200MeV electron moving head-on gives s=me²+2Enu(Ee+pe), whose kinematic boundary occurs atEnu≈13.9543MeV. This is an independent four-momentum illustration only; plasma occupation/blocking and a nonzero rate require the full medium integral.

## Release limits

The original contract clearly labels the old NMB staging/source-access limitation. This review did not freshly fetch/compare the pinned private production cooling source, and makes no new parity claim. The existing source-labelled cold electron/muonDU and pairing prescriptions should remain intact until matched forward comparisons are available.

Implementation must still define zero-baryon-species endpoint handling whenYp=0 or1, because the present positive-density free-gas root API does not itself implement those exact endpoints. That is a future API/domain gate, not an inconsistency in the research contract's physical charge domain. No executable weak API is proposed as accepted by this review.
