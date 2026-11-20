# Independent fixed-bath lepton-scattering moment-fit review

**Accept the optional positive-moment fits only at the measured fixed-bath scope. Reject any signed-energy accuracy/sign qualification.** This review has114 passing selected checks and a NEW preserved signed-output sign failure, in addition to the author's two already excluded signedRT failures. The API correctly exposes a global False signed qualification flag and advises detailed quadrature. No coefficient, author source, original failure or gate was changed.

At a fresh adaptive post-fit diagnostic **E8.14MeV**, the fitted neutrino energy-change moment is **+0.06886595010MeV/s**, whereas actual CM quadrature gives **−0.11647091575MeV/s** and a separate phase-space integral gives **−0.11647052316MeV/s**. Thus using the diagnostic signed fit would infer material cooling where the detailed result gives material heating. Its positive moment fits and even the declared absoluteRT gate still pass near this zero. This is not a new blocker to the already limited positive-moment contract; it demonstrates why the explicit signed rejection/fallback instruction matters. The new sign test is an adaptive diagnostic, not a preregistered holdout or a reason to retune this frozen model.

Reviewed `api.py` SHA256:335f1acfdf3ff122a66b4afd5711b956dc7b3083e97d03d012742846c7eb2182; author manifest2d942a4dd968b34058eef7854d5d19c5196993a870f0ede73c6ec154a364ce20. Exact model/API/registration/lock/qualification snapshots and every original artifact hash are saved. All author hashes remain unchanged after actual numerical probes.

## Registration, selection and reproducibility

The12 training energies and6 heldouts exactly match their recorded registration. Every heldout is inside its declared blocked interval, and no training energy is in any blocked interval. Original training and heldout quadrature reports contain no ignored source-refinement failure. Model, registration and training-data hashes agree with FITTING_LOCK, which records heldout-file absence before model lock. Source inspection confirms the builder reads training references for coefficient/degree selection and does not read heldout moment values. This corroborates the recorded training-only development sequence; a saved absence flag by itself is not a cryptographic proof of all prior private knowledge.

An independent **QR** solve of the Chebyshev design matrix reproduces every candidate degree2..7 and its leave-one-training-out error, without invoking the author's `chebfit` selection code. The smallest candidates meeting both training gates are degrees3,4,2, hence12 coefficients. The saved positive qualification explicitly denies a uniform-domain error certificate. No validation coordinate is reused to choose a new coefficient. The two signed training failures at19 and30MeV are recomputed from the unchanged model and saved detailed data, and agree with the exact failed values in both validation and QUALIFICATION. The API returns its False signed flag at both energies.

The fit is exp(Chebyshev(log E)) for R, Rtr and Qout in their stated numerical units; DeltaQ=Qout−E R is derived by subtraction. This preserves positivity and, in the sampled API interval,0<=Rtr<=2R. It does not preserve the small residual's relative precision or sign. No AD derivative fidelity is established. This is an optional scalar-moment convenience with a detailed-physics path, not an EOS, redistribution kernel or substitute for the physical AD chain.

## Fresh actual fallback sources and fit errors

At two NEW energies8.19 and23.17MeV, `method='detailed'` invokes the actual pinned two-target CM integrator, with physical FD e−/e+ targets, final charged-lepton holes and **unblocked final neutrinos**. These energies are absent from the saved training/heldout lists. A separately called coarse physical rule agrees with each actual fine fallback at the original1e−4 positive and signedRT source gates. Model inspection verifies the fallback imports the physical adapter and calls its detailed function; it does not select a cached data row, interpolate saved moments or evaluate the coefficient fit. Its source and model hashes are checked before the call.

| E[MeV] | Actual R[s^-1] | Actual Rtr[s^-1] | Actual Qout[MeVs^-1] | Actual DeltaQ[MeVs^-1] |
|---|---:|---:|---:|---:|
|8.19|10.2285094304|7.27244536833|83.4253094414|−.346182793840|
|23.17|29.8255583005|13.5924037502|490.595385908|−200.462799915|

Maximum positive-fit error at these fresh points is.168252%, below the unchanged1% gate. Their signedRT diagnostic errors are+.00901727 and−.0214698, respectively, below.03. But at8.19 the derived signed fit is−.161716402427, already about53% away from the small detailed signed moment. A small absoluteRT error is not a relative/sign certificate near a zero. The additional8.14MeV counterexample at the top makes this failure concrete, with the global signed rejection flag still present.

## Independent moment/phase comparison near the zero

The separate `phase_probe.py` computes no fit value and calls neither the author's CM integrator, differential evaluator nor transverse-shell phase reference. It integrates the primary accepted invariant polynomial analytically over target azimuth, adaptively over **target energy**, using direct Gauss cosine and a **sinh(Eout−E)** outgoing-energy representation. It includes both actual target charge populations and holes; the signed moment is accumulated as (Eout−E)dGamma directly rather than subtracting two final positive moments.

At8.19MeV, controls18energy/24cosine/100MeV/65T target-tail versus26/36/140MeV/90T pass1e−4 positive and absoluteRT refinement gates. Fine values are R10.2285094464,Rtr7.27244536212,Qout83.4253099792,DeltaQ−.346182386750. Against actual fine CM, positive discrepancies are at most about6.5e−9 relative and signed error about2.0e−8 in RT units. Direct outgoing100..140MeV tail integrals are negligible at1e−8 scaled gates. At8.14, the same independently refined phase calculation confirms the negative signed value before judging the fit's positive sign.

This is an independent coordinate/quadrature implementation of the same declared free Fermi-current model, not independent empirical physics or a weak-uncertainty estimate. The reviewer authored the earlier elementary scattering leaf and openly reuses its accepted source conventions. The new phase reduction differs from both that CM implementation and the fit author's transverse-shell integration; none of those shared physical assumptions is validated merely by agreement.

## Domain, closures and interpretation

The API rejects energy outside1..30MeV, nonfinite/Boolean/complex energies, any other T or mu_e, antineutrino flavor, an incomplete target set, outgoing FD radiation and unknown method choices. This strict fixed-state contract is appropriate for the fitted source. No T/density/composition interpolation, muon target or all-flavor inventory is supplied. Both positive fits and the detailed API retain the same physical bounds; the API does not silently offer an out-of-domain fit or change closure when asked for detail.

GF, electron rest mass, illustrative sin²thetaW and the CC+NC electron-flavor coefficients are fixed source inputs, not fitted calibration parameters. There is no Vud factor, and the physical final-neutrino phase measure is already included in the moments; adding another DOS would be wrong. R is scattering-event frequency and Rtr is its1−cos_lab weighted moment, not an absorption coefficient or general inelastic diffusion closure. The signed moment is per incident neutrino; bath energy change is its negative. Neither its zero nor a moment fit defines full radiation LTE.

Existing source limitations, nuclear/plasma omissions, weak-parameter uncertainty and final-neutrino blocking choices remain unchanged. Six blocked points plus three new diagnostic energies are not a continuous1% error theorem. The mathematical501-sample positivity checks in the author API evidence are not501 physical source points. No timing/production-performance, derivative, evolving matter, full transport or hydro claim is added.

## Evidence and qualification boundary

`audit.py` gives89/89 selected provenance/QR/control/fresh-source checks in20.56s. `phase_probe.py` gives12/12 independent phase/refinement/tail checks (individual runs about.03–1.10s). `sign_probe.py` gives13/13 source/fit-contract checks and separately preserves the NEW failed sign diagnostic; it does not reclassify that diagnostic as an admitted pass. The original two signedRT failures remain byte-for-byte in the pinned source evidence and are explicitly reproduced. No gate was relaxed, heldout added to fitting or author patched.

Admit R,Rtr,Qout only as optional measured fixed-state conveniences with detailed fallback. Do not admit a signed heating/cooling fit on this evidence. A future joint positive/signed-moment construction must predeclare sign/absolute cancellation metrics and a new independent data split; these now-known failure and holdout points cannot be presented as unseen validation after retuning.
