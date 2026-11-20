# Independent review: optional free EC/PC moment corrections

**Selected numerical fit evidence ACCEPTED; runtime module admission BLOCKED by two qualification-guard failures.** Review checks:191 PASS,2 admission FAIL, no new physical moment-fit failure. The two failures expose the same unpinned/incompletely checked qualification-report boundary. Actual frozen training/heldout data and coefficients remain intact. No author/upstream code or data was modified; no retuning occurred.

Reviewed `approximation.py` SHA256`bd7480c670ca7a00b336c234e0d825b38701f26620f4065f719b30893acb91ee`, manifest`db256f83bf4840d5a3a5985ef22ee993642dc17fc7faa4f97db165968d95f1f4`, coefficient freeze`6e81f38d4538732abc49462331c7277160ba220518747f43aca1b13421d71a12`.

## Fresh validation lock and physical forward evidence

`VALIDATION_LOCK.json` was written at2026-10-16T01:48:28.693826Z BEFORE any fresh reference calculation. It preregisters four states, separate positive-output1% gates,3e-6 detailed refinement and3e-8 target-density gates, with no retuning. All four points are different from the11 training and8 author-heldout states.

| T MeV,nB fm^-3,Ye | Actual total mu_e/T | Largest absolute individual fit fraction error |
|---|---:|---:|
|1.6,4e-7,.32|.6967519233|5.5907963e-5|
|2.8,8e-7,.23|.1934499625|1.0933684e-5|
|1.87,7e-7,.41|.9384944455|2.7413111e-5|
|2.37,3e-7,.27|.1409630007|7.6563975e-6|

At each state I independently reconstruct baryon total chemical potentials using the relativistic particle FD Bessel/fugacity series and solve electron-minus-positron charge by original-momentum Gauss quadrature. The fitted EOS metadata agrees with the physical inputs; all new points satisfy |mu_e/T|<=3 and baryon eta<=-5. This is the intersection admission, not permission to use the whole nominal box.

Actual EC and PC event integrals are executed at BOTH original profiles on these independently reconstructed baths: EC20/32/8/8/8 tail50 ->28/48/12/12/12 tail60; PC28/44/10 ->32/52/12 with tail60. All four separate R_EC,Q_EC,R_PC,Q_PC quantities refine, maximum4.51e-10 versus3e-6. Incoming proton/neutron density integrals pass3e-8. No fixed saved moments, target rescaling or stationary-target substitute supplies the detailed references. The batch takes15.34s. Raw full source outputs and partial checkpoints are retained.

The new16 per-output1% gates all pass, maximum fractional error5.59e-5, or0.00559%. These are fresh selected observations. They do not turn the original author heldout maximum6.87e-5 into a uniform error bound, certify unsampled boundaries, measure physical-model uncertainty or qualify derivatives.

## Fit arithmetic, normalization and leakage audit

The fit is an optional correction to the accepted physical stationary allowed integrals. For each separate positive output it computes allowed*exp(basis dot coefficients). I independently reproduce the allowed baseline using charged-lepton rapidity quadrature, with the accepted same GF/Vud/gA/me/Q and actual signed electron chemical potential. Independent baseline-plus-coefficient arithmetic agrees with the API within1.20e-14. The prior independently accepted allowed assessment and its20 model-accuracy failures remain untouched; these corrections do not retroactively make that approximation accurate.

The basis is exactly [1,T,T²,eta_e,eta_e²,T*eta_e,(nB/1e-6)*T^-1.5], with numerical T in MeV and nB in fm^-3. There are7 coefficients per output,28 total. A separately assembled training-only matrix and QR solve reproduce the frozen least-squares coefficients within7.16e-17; rank is7. This is reproduction of the locked model, not a new fit used by the runtime. No heldout value enters that solve. All11 training file hashes match the coefficient freeze. Saved8-heldout arithmetic and each of32 separate1% gates reproduce.

The original12-point LOCK and preflight failure preserve the rejected T1,nB1e-6,Ye.5 corner (actual mu_e/T~4.1975). REVISED_LOCK2 removes only that point, retains the7 basis functions,8 heldouts and gates, and records the parent-authorized change before any rate data. Rank admission records an empty rate inventory; coefficient freezing refuses pre-existing heldout files and records an empty heldout inventory. Recorded chronology/code is consistent with the stated pre-data revision and training-only freeze. This is an audit of preserved experiment records, not an independently trusted timestamp service. No failed state was deleted after fitting; the original corner still rejects through the runtime EOS admission.

EC is p+e- -> n+nue; PC is n+e+ -> p+antinue. Separate number sources are fm^-3 s^-1 and neutrino energy sources MeV fm^-3 s^-1. Electron/positron occupations and targets are distinct, with physical free particle-only baryons, pair-equilibrated leptons, no muons and empty incoming radiation. Full moving-target/recoil/Pauli/current/form-factor physics is inherited from the already qualified detailed kernels. No new independent current-model validation is claimed simply from executing their two quadratures. No extra target spin or neutrino DOS multiplies the returned event moments.

## Required qualification-consumer repair

`approximate_moments` verifies only `qualified is True` and the coefficient hash in HELDOUT_ASSESSMENT.json. It does not pin that report's bytes or establish complete passing heldout contents. `guards.py` changes only the in-memory file read, without modifying author files, and reproduces two failures:

1. A report retaining the correct coefficient hash and qualified=True but replacing checks with an explicit failed50%-error gate is ACCEPTED.
2. A fabricated two-field report containing only qualified=True and the correct coefficient hash, with no checks or rows, is ACCEPTED.

Both return positive approximations with qualification language despite missing/failed evidence. The exact supplied reports are saved in `guard_results.json`. This blocks the runtime's asserted locked qualification boundary until a narrow author repair and separate recheck. It does not invalidate the current immutable physical heldouts or the new four-point fit evidence. This is a scientific admission/metadata issue, not a claim about arbitrary hostile Python or filesystem security.

The other16 guard checks pass: invalid/outside-box inputs and the EOS-rejected corner, plus false/nonliteral qualification and wrong coefficient association rejected before a physical-preparation spy is reached. All current source pins match; coefficients, original/revised locks and full author bytes are unchanged after experiments.

## Scope and remaining limits

Keep detailed physical quadrature as reference/default and these fitted four moments as an explicit optional convenience only. No composed net source, kinetic root, beta moment, opacity/redistribution kernel, radiation closure variation, interacting EOS-family extrapolation, parameter derivative, uniform sign/accuracy certificate or physical uncertainty follows. Small independent channel errors can amplify near number balance, so this review does not authorize using fitted outputs to replace that solver. No surrogate replaces the physical chain or supplies new reference data.

Commands from Genesis with existing interpreter:

```
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/review_free_capture_moment_fit/probe.py
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/review_free_capture_moment_fit/guards.py
```

The physical/arithmetic/provenance script has175 PASS; guards have16 PASS/2 FAIL. No numerical gate was relaxed, no source fix or dependency was added, and no production/paper/git action occurred. This initial failed module-admission record must remain immutable through repair. The current fit is acceptable evidence for publication with its selected numerical qualifications; the guarded runtime module is not yet admitted.
