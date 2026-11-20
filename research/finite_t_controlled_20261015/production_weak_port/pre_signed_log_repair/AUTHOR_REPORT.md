# Staged free charged-current CPU reference

**FROZEN candidate for DIFFERENT physics/numerics and packaging review; not merge/release qualified.** Exactly eleven staged paths provide detailed physical free EC, PC and neutron-decay/inverse TOTAL moments in an owned real NMB package mirror. Validation:53 pytest tests PASS (40new,13unchanged cold/unit),1505 related raw source/refinement/ledger/provenance checks PASS,13 separate vacuum/proper-time/Doppler checks PASS. These counts repeat moments/states/metadata and are not independent physical models or experimental benchmarks. No new physical numerical gate failed. All existing campaign cold/coarse/model/admission failures remain historical failures; none were retuned or relabelled.

## Implemented scope

Explicit import `nmbackend.reference.weak_reference` exposes immutable FreeElectronBath, Current, Radiation, Controls and MomentResult records, positive/signed log representation and `channel_moments`. EC=p+e−↔n+nue, PC=n+e+↔p+antinue, NEUTRON_DECAY=n↔p+e−+antinue, forward meaning emission. Both F/B occupation products integrate directly; KMS is checked after construction, never used to manufacture an inverse. Net moments use stable local expm1/log-signed accumulation. Number, neutrino/charged-lepton/proton/neutron energies and direct material loss are returned. Empty incoming radiation is exact; common-T FD nue and anti potentials are independent. Integer particle-count stoichiometry, E/Le/charge/baryon and direct common-T entropy checks use positive gross scales, not939MeV rest energy or cancelled net sources.

Bath inputs are explicit TOTAL chemical potentials; no EOS, neutrality/beta solve, density rescaling, fitted rate, interpolation, cache or unmerged utility dependency. Particle baryons, opposite e± chemical potentials and absent muons are explicit. The fixed free physical MN939.56542052,MP938.27208816,ME.51099895069 and full source Guo current (GF1.1663785e−11,Vud.974,gA1.2723,F2=3.706,GP,MV840,MA1000,mpi139.57) use physical q² and bare Mweak=(MN+MP)/2. qGamma=pp−pn, reciprocal adjoint with opposite oriented argument and future decay Q=−qGamma follow the repaired accepted contract. No empirical lifetime calibration, extraQ, mean field, radiative/Coulomb or many-body correction.

The source-labelled runtime scalar invariant reduction is ported from the author EC/primary Guo formula with crossed signs and independently gated against literal u/v sums and Dirac traces. It does not import/copy the independent reviewer PC quadrature as package runtime. Physical two-body moving-target phase and exact moving-neutron three-body decay maps include actual bath-frame blocking. Covariant avg4 capture versus avg2 parent normalization, spin-two initial measures, time dilation, HBAR/HC conversions and no extra neutrino DOS are explicit.

Stage A has NO spectral j/chi, opacity, transport coefficient, EOS, source evolution, photon closure or AD; requested spectral coefficients raise NotImplementedError. No full Urca/flavor/interacting/paired inventory or physical uncertainty claim. The eleven paths are five modules, two tests, one INTERNAL saved regression fixture, example, API/model document and narrow README addition. Existing cold source/API/exports remain exact.

## Preregistered numerical evidence

LOCK.json f65f8e9bb59ae17cfb535a9d72ada558bbc0ec2485f929c2f847ab91ea9968d5 preceded ANY candidate validation. Model/state/current/units, request bounds, profiles and unchanged gates are locked. Source instructions were freshly checked at eight owning paths (absent404); actual main was freshly resolved to ffb5cc72949f2e5401251847aef4c9c7a98dde78. Existing README/minimum dependency declarations were read. An owned package copy contains71 exact baseline blobs plus the eleven-file overlay. No original production checkout, prior PR/candidate mirror, frozen campaign scientific source/review, Git/branch/PR/CI or dependency was changed.

Request T1–3MeV,baryon eta[−100,−5],|mu_e/T|<=3,common-T radiation|mu/T|<=10; orders4–64,tail30–100T,budget<=120s. A separate PREVALIDATION_RESOURCE_POLICY.json, written before validation, limits actual vectorized event arrays to250000 per radial node. These are admission/resource restrictions, not a uniform numerical-accuracy domain.

Before thermal moments, current-only9tests passed; the final suite adds a zero-transfer point-VA/magnetic-transversality/pseudoscalar contraction control. Explicit absolute target-rest sigma [cm²] agrees with independently source-defined Dirac trace and separate angular refinement; vacuum beta exact pair/electron Dalitz rules and old source normalization, actual unblocked thermal rate `<MN/En>*Gamma_rest` and Doppler energy normalization pass. Bare tree rest rate=.001053384620918292s−1; no measured lifetime used.

The three SAVED exactT2,nB1e−7,Ye.3 matched-neutral chemical-bath spectrum-reconstructed total groups supply empty/LTE/nonequilibrium positive F/B R/Q regressions. This is saved research evidence, not a new EOS or external table. Two NEW nonneutral explicit-chemical holdouts were locked before computing:

- T1.6,mun=MN−11T,mup=MP−10.4T,mue=.42T;
- T2.7,mun=MN−10.2T,mup=MP−9.8T,mue=−.3T.

At each heldout, actual candidate base/fine full F/B species-energy moments cover empty,LTE,emission- and inverse-dominated FD radiation. Fresh original source computations compare all six moments; the independent PC momentum/CM-axis oracle covers empty and LTE (not arbitrary changed-affinity closures). The latter method shares the physical Guo polynomial model; comparison is phase/numerical agreement, not physical-model uncertainty. EC and beta source regressions intentionally reuse ported source mathematics, with separate literal-current/phase-coordinate anchors and DIFFERENT review still required.

| Selected check | Largest measured relative or gross-scaled error | Locked gate |
|---|---:|---:|
| saved positive F/B number/energy totals |2.124e−8|3e−6|
| all-six-moment base/fine refinement |6.875e−9|3e−6|
| fresh source all-six-moment agreement |4.152e−11|3e−6|
| isolated fine-rule tail increase |2.132e−14|3e−6|
| rest-inclusive species energy ledger/grossQ |4.893e−12|3e−9|
| direct event-loss energy ledger/grossQ |8.878e−14|3e−9|
| direct FD entropy versus affinity/grossR |2.180e−14|3e−9|

Isolated tails at the second bath test empty and both affinity directions, at unchanged fine-axis controls; beta uses the finite direct-p rule, not a meaningless Laguerre-tail parameter change. Every scientific process remained below120s; longest recorded collection is 20.204s (candidate_1_NEUTRON_DECAY.json). Check actual stdout/raw for exact durations. No expensive grid scan or installs.

## Representation, provenance and retained drafts

Finite positive logs survive ordinary under/overflow with value=None; exactzero inverse is permitted only for the absent incoming occupation. The integrated finite-T forward/FD-inverse branch cannot be replaced by falseempty. Constructors reject Boolean/nonfinite/huge numerics, malformed positive/signed quantities and invalid controls. Timeout raises without a complete result. LTE net cancellation, entropy and energy use positive gross scales. None diagnostics mean unrepresented, not PASS. Strict guards/timeout/incomplete/immutability and spectral-unavailable tests pass; this is an in-process scientific API, not an untrusted-code sandbox.

During drafting, inspection strengthened SignedQuantity ordinary-value validation to reject Boolean values. No physical failure was observed and numerical equations were unchanged. Exact old quadrature source and all prechange actual rows are preserved under pre_signed_value_guard/. All current-source forward rows were recalculated; five overlapping full profile groups are byte-identical old/new. No prior-source result is presented as a current-source execution: final candidate row files fingerprint all five actual physical modules. A driver-only reference branch was subsequently changed to use the bounded independently accepted PC scalar/phase oracle rather than a heavy explicit-spinor total; no package runtime changed. The current numerical comparison scope explicitly names that reference.

All60 locked scientific/evidence files remain unchanged. verification.json proves actual installed-style module origins inside the owned real package, baseline71blob identity, exact staged/mirror bytes, all candidate-row source fingerprints, no campaign/generic-loader/import-time IO dependency, no unmerged lepton/baryon/RMF dependency and static Python3.9 grammar. No unsupported SciPy log_expit import is used (private NumPy logaddexp identity). Executed environment Python3.12.12,NumPy2.3.5,SciPy1.15.2,JAX.10.1,pytest9.0.3. NumPy2.3.5 is outside the separate validation pin<2.3. Minimum Python3.9/NumPy1.21/SciPy1.7 runtime, normal full CI, clean installation, wheel/package review, AD and external/empirical uncertainty qualification remain OPEN. Different numerical/physics review and different packaging review precede any draft PR/publication decision by root. No broad phase or source-binding authorization follows from these author checks.
