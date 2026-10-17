# Joint positive/signed lepton-scattering moment fit, version 2

**Author-frozen research prototype; independent review pending.** This new joint representation passes the preregistered gates at eight fresh validation energies. It preserves all version-1 coefficients, source data, signed failures and reviewer wrong-sign counterexample. This is an optional fixed-state convenience with actual detailed quadrature near cancellation, not a replacement for the physical chain, a spectral kernel or a uniform accuracy/sign certificate.

## Physical scope and units

The physical model is finite-mass νe scattering from both free FD e− and e+ targets at T=2 MeV and total electron chemical potential mu_e=.17446333200826397 MeV. Outgoing neutrino radiation is unblocked. Spin-two target measures, target holes, one incident active neutrino helicity and electron-flavor CC+NC interference follow the frozen detailed source and its prior independent review. GF=1.1663785e−11 MeV^-2, me=.51099895069 MeV, sin²thetaW=.23129, CV=.96258, CA=.5; there is no Vud factor. QED/electroweak corrections, correlated matter and uncertainty in the physical current model are not estimated by fit residuals.

Independent momentum integrals give n_e−=2.03428636784328e−7 and n_e+=1.73428636784328e−7 fm^-3. Their net charge is 3e−8 fm^-3, consistent with the earlier nB=1e−7,Ye=.3 material bath. Neither total pair population is replaced with net charge. This API takes the fixed T/mu state, not an independently varied baryonic composition.

R=∫dΓ and Rtransport=∫(1−cosθ)dΓ have units s^-1 per incident νe. Qout=∫Eout dΓ and DeltaQ=∫(Eout−Ein)dΓ have units MeV s^-1 per incident νe. Positive DeltaQ heats neutrino radiation and removes the same energy from the prescribed material bath. Number scattering conserves neutrino number and material electric charge. These integrated moments cannot reconstruct an angular/energy redistribution kernel or a Pauli-blocked final-radiation rate.

## Preregistration and joint construction

`PREREGISTRATION.json` was written before new calculations. ALL old v1 training, heldouts, zero diagnostics and reviewer8.14/8.19/23.17 data were declared known development. The new training set contains the 18 old qualified references plus actual recomputations at8.14,8.19,23.17. These 21 development energies select coefficients without any fresh validation result.

Let x=2log(E/1MeV)/log30−1 and T=2 MeV. Fit logR, logRtransport and the SIGNED dimensionless g=DeltaQ/(RT) with separate Chebyshev expansions. Derive Qout=R(E+Tg) and DeltaQ=RTg from the same R,g. There is no logabs across the sign zero. Both positive moment bounds and E+Tg>0 are checked at every call; 4001 development-domain samples satisfy these bounds. This sampling is not a proof between points.

Degrees2–8 were considered. For positive moments, training/leave-one-out gates were .002/.005 relative. For g, gates were .0005/.002 absolute. The smallest degree passing both gates was selected: **R degree4, Rtransport degree5, g degree6**, 18 coefficients total. Maximum training/LOO errors are .001225/.004255 for R, .000861/.002114 for Rtransport and .0002593/.001597 for g. Coefficients and all candidates are in `model.json`; its SHA256 is `f881ee6f9733fef2120fe5c75cb38f7a8e44bd85f88bba4d3d80a829c690a6f0`. `FITTING_LOCK.json` records the exact model/training/preregistration hashes and absence of fresh validation evidence at locking. The model was never changed afterward.

The NEW preregistered validation energies were1.37,3.63,6.47,8.161,8.227,11.83,17.41,26.39 MeV. Validation does not select degrees or retune coefficients.

## Measured evidence and explicit cancellation fallback

All84 development quadrature gates pass. On the eight fresh validation points, all80 gates pass: actual CM coarse-to-fine refinement, positive fit errors, signed absolute error, signed relative error away from zero, and actual-fallback checks. Positive gates are1%; signed error is bounded at tested points by .005 Rref T, with an additional2% relative gate only where |DeltaQref/(Rref T)|>.25.

| Fresh validation quantity | Maximum measured error |
|---|---:|
| R relative | .11931% |
| Rtransport relative | .07081% |
| Qout relative | .12608% |
| DeltaQ absolute / (Rref T) | .00154874 |
| DeltaQ relative away from zero | .13846% |
| Positive reference refinement | 2.35e−8 relative |
| Signed reference refinement | 2.55e−9 RT |

Default `evaluate(E,method='auto')` calls actual fine CM quadrature whenever |gfit|≤.08, or fitted physical bounds fail. This is a declared conservative diagnostic guard, not a certified bound on unknown fit error. It uses no saved reference-row cache. Method `detailed` requests fresh quadrature at any supported energy. Method `diagnostic_fit` exposes the raw fit with an explicit guard flag; it is intended for assessing residuals and does not evade the default fallback.

Both fresh cancellation-region points trigger actual fallback. At8.161 MeV, true g=−.010430821912 and DeltaQ=−.212591605479 MeV s^-1; at8.227, true g=−.0252051298183 and DeltaQ=−.518063374435 MeV s^-1. The raw locked fit also has the correct signs at these points. A distinct analytic-azimuth/transverse-shell D integral at8.161, outer/angular orders12/20 and16/28, upperEout160 MeV and target tail90T, passes12 gates. Its fine agreement with CM is ≤1.51e−9 relative for positive moments and5.52e−11 RT for signed transfer. The nonzero negative transfer there is resolved independently of the fit. No uniform zero location or sign certificate follows.

The API guard/fixed-state/math suite has624 PASS, including a NEW explicit actual fallback at5.731 MeV. These are primarily mathematical/admission checks, not624 independent physical benchmarks. A separate75-check report covers independent pair populations, physical source pins and unchanged v1 manifest bytes. There are no new failed gates to suppress or loosen. Previously failed v1 gates remain in their original leaf and were not reclassified as passes.

## Commands, cost and qualifications

Run in order with the existing `/Users/rossma/miniconda3/bin/python`: `develop.py`, `fit.py`, `validate.py`, `anchor.py`, `check_contract.py`, `final_checks.py`, then `make_figure.py`. Frozen collection/fit/validation scripts refuse overwrite where applicable; use a separate authorized output copy for replay. Each detailed incident-state experiment stayed well below120s. Fresh validation collection took101.09s total, maximum19.33s per incident energy including a repeated fresh fallback; the independent phase calculations took.668 and1.257s. The API's additional physical fallback took one ordinary fine detailed call. Numerical quadrature errors are far below the declared fit budgets at the tested points.

`joint_moments.png/pdf` shows the fixed-state signed reference, positive validation residuals and signed RT residuals; red squares mark actual fallback points. The image was visually checked. Its connected residual lines do not certify intermediate energies. New heldout gates, dense physical bounds and quadrature convergence provide selected evidence only. Domain remains strictly E1–30 MeV, this T/mu/flavor/both-target/unblocked closure. No extrapolation, arbitrary radiation blocking, derivative/AD qualification, full-state parameter fit, continuum transport certificate or physical model-uncertainty estimate is provided. Independent review must qualify the new model and actual fallback before use in broader campaign results.
