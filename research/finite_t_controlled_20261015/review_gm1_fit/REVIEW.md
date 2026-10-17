# Independent review: restricted one-potential GM1 fit

**Accept the sampled bulk-potential contract, with an explicit low-T affinity exclusion.** The review records101 passing checks and1 preserved failed check out of102. The main raw report remains `passed:false`; the failed1% exponential-balance-factor gate is not loosened or relabeled. No author/refit/upstream change was made. External matched finite-T GM1 validation remainsOPEN.

## Independent derivatives and fresh exact states

New coordinates, absent from the author's training and final heldout tuples, are(nB,T,Yp)=(.2537,2.73,.3237),(.1811,.12,.1817),(.2593,2.99,.3291). Exact field/density constraints are checked before comparisons, including perturbed states. The local exact canonical scalar curvature and full fitted/exact chemical Hessians are positive at all three states. These are nuclear-only local branch checks, not phase coexistence or a global density rectangle.

The derivative implementation is independently checked using a **value-only manual Chebyshev recurrence**, not the author's `chebder` or analytic Y-chain-rule/Hessian formulas. Complex-step derivatives of f(nn,np,T) reconstruct both chemical potentials; complex-step density/T derivatives reconstruct pressure and entropy. Finite differences of those independently computed complex-step chemical potentials reconstruct the full two-charge Hessian at two decreasing density steps. Maximum relative Hessian discrepancy against the fitted analytic implementation is1.47e-9.

Separately, re-solved exact GM1 chemical potentials at independently changednn ornp reconstruct the exact Hessian. That finite-difference matrix agrees with the frozen implicit-response Hessian within4.19e-10 relative. The fit-to-reference Hessian errors below therefore do not rely solely on its author's implicit-Hessian implementation. Fresh entropy differences at two decreasing temperature steps also check exactCv and the fit; all12Cv gates pass, with maximum fittedCv error0.02873%.

| New state(n,T,Y) | Free-energy error [MeV/baryon] | Chemical-difference error [MeV] | Exponential balance-factor error | FullH error |
|---|---:|---:|---:|---:|
|(.2537,2.73,.3237)|5.74e-5|+.00324606|0.11897%|0.07516%|
|(.1811,.12,.1817)|5.31e-6|−.00185976|**1.53785% FAIL**|0.08035%|
|(.2593,2.99,.3291)|1.61e-4|+.00381454|0.12766%|0.10984%|

Across these new points, the largest energy error is0.00031266MeV/baryon, pressure error0.01805%, entropy error0.02757%, and individual chemical-potential error0.009278MeV. All pass the unchanged sampled bulk gates. These fresh maxima modestly extend some author-heldout maxima; the published author table is not a continuous-domain bound.

## Preserved low-temperature failure

TheT=.12 point lies inside the bulk interface domain but below the author's final warm heldout temperatures(T>=.8). Its fitted mu_n−mu_p error is−.00185975777538MeV. Consequently|exp(delta_mu/T)−1|=.0153785057539, exceeding the original1% diagnostic gate. The tiny free-energy and acceptable bulk derivative errors do not prevent this amplification. The supplied potential can retain its selected bulk use, but uniform low-T affinity/balance-factor accuracy is **excluded**.

This factor is a sensitivity diagnostic only; neither passing nor failing it establishes a physical weak-rate error. Dispersion relations, effective masses, currents and response functions are not supplied by this fitted potential. Inserting its chemical potentials into an unchanged microscopic kernel can violate consistency. AsT→0, any nonzero chemical-difference error can be exponentially amplified; theT=0 entropy/Cv limits cannot fix that problem. No repair/refit is performed in this review.

## Data separation, guards and provenance

`data_audit.py` reconstructs all60 frozen coefficients from the **training targets alone**, using a separately assembled recurrence matrix and direct linear solve. Largest coefficient differences are3.47e-14(cold),3.85e-17(T²),1.54e-20(T⁴). The matrix condition number143.892 matches the reported value. Final heldout tuples are disjoint from the old diagnostic tuples and from the training states. The code's numerical fit does not use final heldout targets.

That audit does not prove a historical predeclaration or erase model selection: the author explicitly disclosed that older diagnostic holdouts motivated the isospin-even basis. They remain model-selection evidence. The frozen new points chosen in this review provide additional tests, and their cold failure is preserved. Failed low-order baseline results remain unchanged.

TheT=0/3 andn/Y endpoint evaluations are finite; cold entropy/Cv vanish exactly. Out-of-domain/nonfinite inputs are rejected. These are implementation/domain checks, not exact physical EOS error certificates at every endpoint. All selected author/reference hashes match before and after the physical probe and at review completion.

`probe.py`/`results.json` retain85main checks(84pass/1fail), manual derivatives, exact forward states and Hessians. `cv_probe.py` adds12passing checks; `data_audit.py` adds5. Physical main/Cv work costs4.49s+3.58s. The review preserves its own failed gate without changing any frozen source. Accepted scope is an analytic internal-GM1 **bulk** component at sampled states; production, external physical EOS parity, uniform error bounds, microscopic rates and coupling sensitivities remain unqualified.
