# Optional compact corrections for free EC/PC moments

This leaf builds a small positive approximation alongside the detailed physical chain. The detailed actual FD/full-current event integrals remain reference/default. The approximation returns only separate electron-capture/positron-capture number and neutrino-energy moments. It never returns a composed net source, composition root or derivative. Independent fitted rate errors can amplify strongly near a composition balance; this product is not a replacement for that solver or an AD chain.

## Admission-driven plan revision before data

Original LOCK.json froze12 training states and8 new heldouts before any rates. Actual EOS preflight identified T1MeV,nB1e-6fm^-3,Ye.5 with mu_e/T=4.19754889294, outside the unchanged EC physical event API's |mu_e/T|<=3 admission. The original lock and full preflight failure remain immutable. No guard was relaxed or bypassed.

Before any physical rate or fit data, the parent authorized REVISED_LOCK2.json: remove exactly that rejected training corner, retain the same7 basis functions,4 outputs, all8 heldouts and1% gates. This is an API-admission change before data, not model selection against residuals. The11x7 matrix has rank7, raw condition210.62 and column-normalized condition110.33. All19 revised states pass the original EC temperature/lepton/baryon-fugacity controls with actual matched EOS roots. The candidate domain is T1–3MeV,nB1e-7–1e-6fm^-3,Ye.2–.5 INTERSECT actual EOS |mu_e/T|<=3 and baryon eta<=−5; it is not the full rectangle. Passing selected heldouts does not certify a uniform error even on that intersection.

Training states are T{1,2,3} x nB{1e-7,1e-6} x Ye{.2,.5}, minus the one rejected corner. New heldouts are T{1.35,2.65} x nB{2e-7,6e-7} x Ye{.3,.45}. All initial and revised admissions, exact source hashes and timestamps are retained.

## Physical targets and baseline

Every state has fresh actual free relativistic particle-only n,p density roots and neutral pair-equilibrated e±, physical total chemical potentials and rest energies, no muons, and empty incoming nue/antinue. EC is p+e−→n+nue; PC is n+e+→p+antinue. Actual moving target FD distributions and final baryon/electromagnetic Pauli factors are included. Full tree current, reciprocal crossing, weak magnetism, induced pseudoscalar, q² form factors and bare weak mass are exactly those of the previously reviewed source kernels. No measured-lifetime normalization, Coulomb/radiative or interacting-medium correction is added.

Both existing event quadrature profiles are executed at every training AND heldout state, retaining full raw outputs. Coarse/fine EC controls: proton20/28, charged-lepton32/48, initial/outgoing angles8/12, azimuth8/12, tails50/60. PC neutron28/32, positron44/52, angular10/12, tail60 at both. Actual target-density integrals require3e-8 and each of4 number/energy outputs separately requires3e-6 refinement before admission. These two-profile checks are numerical evidence at those states; they share physical source implementations and do not measure model uncertainty. No saved detailed moments are used to scale new states.

The baseline is the unchanged weak_rates/static_capture.py physical allowed one-dimensional integral, with matched exact GF,Vud,gA, electron mass and physical Q=Mn−Mp. Target densities and actual EOS FD electron chemical potential are supplied at each state. This stationary, unblocked-nucleon limiting model retains exact-mass leptons and beta electron blocking but omits recoil, nucleon motion/Pauli factors, weak magnetism/pseudoscalar/q² corrections. The separate allowed_rate_assessment leaf measures those combined differences; its independent review determines that assessment's acceptance. The current fit does not assign an error to individual missing mechanisms.

## Locked approximation

For each of R_EC,Q_EC,R_PC,Q_PC, fit log(full/allowed) in exactly

    [1,T,T²,eta_e,eta_e²,T eta_e,(nB/1e-6) T^(-1.5)], eta_e=mu_e/T.

T is numerical MeV and nB numerical fm^-3. This gives7 coefficients per output,28 total. Unweighted linear least squares uses training column L2 scaling for numerical conditioning and transforms back to the original locked basis. Positivity follows from allowed_moment*exp(basis·coefficient). No coefficient/basis fitting uses heldouts, and no basis revision after heldout failure is permitted. Coefficients, training raw hashes, rank, singular values, residuals and an empty heldout-file inventory are frozen in COEFFICIENT_FREEZE.json before heldout collection.

The public function approximate_moments is explicitly optional and requires finite real scalar inputs, the box plus actual-EOS intersection, unchanged source/coefficient hashes and successful locked heldout assessment. No extrapolation, beta model, parameter-gradient, alternative EOS/current inference, arbitrary neutrino closure or uniform physical-error guarantee is offered. Outputs use fm^-3s^-1 and MeVfm^-3s^-1 for number and neutrino energy. No fitted model is used as reference data for subsequent fitting.

## Frozen measured candidate (different independent review pending)

All19 admitted physical states pass both actual EC/PC profiles; largest individual moment refinement is9.578e-9 against3e-6. Training and heldout collection groups took43.92s and29.83s. Training maximum fractional fit error is4.3324e-5 (0.00433%). The8 locked heldouts pass all32 per-output1% gates; maximum fractional error6.86983e-5 (0.00687%). No basis, coefficient, output weighting or state list was adjusted after examining heldout residuals. These are approximation errors relative to the declared free-current reference, not observational/physical uncertainty. No failures were hidden by summing channels.

The113 API/provenance checks include the original rejected corner, finite/type/extrapolation controls, all8 heldout arithmetic/positivity and exact output names, source/coefficients, all19 actual EOS states, original-versus-revised lock identity and training/heldout separation. The original admission failure remains in preflight.json and STATUS.md; successful locked heldouts do not restore the excluded corner. The original blocked first-plan collector is retained as initial_staged_collect.py.txt.

COEFFICIENT_FREEZE.json SHA2566e81f38d4538732abc49462331c7277160ba220518747f43aca1b13421d71a12 predates any heldout collection; its contents include all11 exact training data hashes and an empty heldout inventory. The runtime API pins that coefficient file and its matching heldout assessment. Rank was recorded before all rate data. These explicit stages let a different reviewer inspect leakage/provenance and recompute a fresh new point independently.

Reproduction order: preflight.py (original immutable block), rank_admission.py (revised pre-data check), run_group.py training, fit_training.py, run_group.py heldout, check_heldout.py, check_api.py. Existing physical row files and coefficient freeze refuse overwrite; full reproduction requires a distinct authorized output copy rather than modifying this frozen record. Running preflight/rank again in place would overwrite timing/provenance, so inspect their saved outputs instead. No detailed source/current/EOS file, production code, paper repository or dependency was changed.

The different independent allowed-rate assessment review has now accepted its selected interpretation (review_allowed_rate_assessment/REVIEW.md SHA2560c232395374f40ea416e6ad87deeea05fdbe60dee91e5a11a63909078d6e652f; exact pin in ALLOWED_ASSESSMENT_ACCEPTANCE.json). Its20 model-accuracy failures remain failures. This provenance update does not independently accept this fit or attribute the combined physical discrepancy to any single correction.
