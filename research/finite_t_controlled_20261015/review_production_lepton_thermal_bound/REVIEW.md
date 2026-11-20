# Independent thermal-hole bound review

**Candidate thermal functional completion remains BLOCKED at four new selected muon states.** The strict omission bound is sound and both original electron failures are repaired. This does not complete the returned-state thermal utility: four preregistered states still raise in the original hole quadrature. No author files were edited and no gate was relaxed.

Frozen candidate source e0c16fc5899d5a9bfe2e8c89ba7255a60e80f0bb620c4f3103a7e4bc68487b90; manifest26c63c1865b6dfcdc13968a78f21ab2a71ff15efd3c6fd22b19e4d52189c8e14. VALIDATION_LOCK.json was produced before candidate inspection/comparisons and while the author was still preparing the candidate. Exact electron/muon threshold-side states were chosen by the independent oracle; the root later communicated rough ranges to the author only after author freeze. This is a source-independent forward/bound check with recorded provenance, not a blinded collaboration claim.

There are66 independent pre-candidate oracle PASS,104 candidate/provenance PASS and95 historical regression PASS (265 passing checks/tests total). Raw candidate results retain6 failing gates: **four distinct functional-completion failures plus two dependent represented-hole checks**, not six independent physical counterexamples. initial_results.json and INITIAL_FAILURE_PIN preserve the report before follow-up localization. Original prior140 PASS/2 electron FAIL and earlier production-port reviews remain unchanged.

## Rigorous bound and what it fixes

For xc>0,v=m/T>=0,C=1/(pi² HC³), the positive physical hole term is

    H=C T^4 ∫0^xc sqrt[x(x+2v)](v+x)(xc−x) vacancy(x) dx.

Using vacancy<=1 and monotonic geometry gives the author's conservative H<=C T^4 sqrt[xc(xc+2v)](v+xc)xc²/2. Each positive finite factor x satisfies x<2^frexp_exponent(x), strictly. The two sums are bounded by the stated extra powers of two, including the v=0 branch. Thus the returned integer/half-integer B rigorously gives H<2^B in the declared physical energy-density units. All exponent sums are exactly representable in float64. If B<=−1075, H is strictly below half the smallest subnormal2^-1074 and rounds to zero. Equality of B to−1075 is safe because the underlying power bounds are strict. This is a bound on a component, not the thermal total, net density or exact physical absence. It introduces no density floor.

My independent tighter bound integrates sqrt(x)(xc−x) exactly, giving H<=C T^4 (4/15)sqrt(xc+2v)(v+xc)xc^(5/2). A separately ordered x=xc*t positive scalar quadrature and original scaled-momentum FD integrals agree with Bessel number/energy/thermal limits; actual FD corrections at these dilute points have an explicit negligible exponential bound. Eight threshold-side states (both masses) were fixed at this tighter bound's log(halfminsubnormal)+[-2,−.2,+.2,+2]. These are not expected to share the author's looser dyadic omission boundary.

At m/T600,mu/T±.53 for electrons, the preserved original source still reproduces its unresolved-quadrature exceptions. The new source skips only a rigorously unrepresented hole and returns the represented thermal total2.411480764183e-274 MeV fm^-3 within3e-10 of independently ordered FD momentum/Bessel expressions. Returned thermodynamic fields remain bitwise identical. All four electron threshold-side states pass, including two actual hole restores5e-324 and2e-323 rather than being incorrectly skipped.

## Four remaining completion counterexamples

All use m=105.6583755MeV and mu/T=.53. Thermodynamic state_mu results are valid and agree with independent momentum/Bessel number and energy. The thermal totals below are normal, not total-heat underflows.

| m/T | T (MeV) | mu (MeV) | independent thermal total (MeV fm^-3) | physical hole ordinary float |
|---|---|---|---|---|
|439.2588688799|.2405378308454|.1274850503481|4.5458895056e-195|0, finite positive log|
|438.1825592793|.2411286648966|.1277981923952|1.3386405842e-194|0, finite positive log|
|437.9433805983|.2412603550615|.1278679881826|1.7017518815e-194|5e-324|
|436.8670820899|.2418547421668|.1281830133484|5.0111960549e-194|2e-323|

All four old AND new thermal_energy_density calls raise ArithmeticError: Unresolved finite-temperature lepton quadrature. diagnostic_results.json traces the exact failure to hole_integrand after successful state moment/response integrals. The dyadic upper exponents are−1074.5,−1072,−1069.5,−1067: the conservative omission gate correctly does not apply. This is not a regression caused by the new bound, but it is a continuing functional-domain gap. A matched old/new exception is not a thermal forward PASS. Two nonzero represented-hole checks fail only because those whole calls fail; they are dependent consequences of the same two corresponding completion failures.

The remaining numerical problem is integration of an extremely small raw dimensionless component with a zero absolute tolerance, requiring an unrepresentable absolute error even where dimensional restoration leaves a representable component. The safe bound cannot be widened to discard represented holes or unresolved totals.

A concrete next numerical rung is the exact coordinate x=xc*t, pulling xc^(5/2), sqrt(xc+2v), (v+xc) and the maximum vacancy scale outside the integral. The unit-interval integrand becomes sqrt(t)*sqrt[(2v+xc*t)/(2v+xc)]*[(v+xc*t)/(v+xc)]*(1−t) times the scaled vacancy. It is nonnegative and O(1), with physical scales restored together in logarithms. For v=0 the normalized polynomial is t²(1−t), integral1/12, an exact independent anchor. This is a suggested algebraic implementation/review target, not a repair made or qualified by this reviewer. Existing quadrature/error guards should remain, with represented near-bound components independently checked.

## Preserved interfaces and regressions

Only the private dyadic helper and the additional hole-branch conjunction differ. All other original function ASTs are identical. Removing that conjunction makes thermal_energy_density's AST identical too. Public signatures/defaults, state fields, constants, target/spin measure and units remain unchanged. Three new warm/degenerate successful states agree old/new. The stable private log-expit helper passes independent scalar tests, and a real package-relative import succeeds when the newer scipy.special.log_expit export is temporarily absent and restored in finally. This remains a missing-export simulation, not an installed minimum-SciPy runtime test. The new math.frexp bound introduces no newer SciPy dependency.

The original pre-hole test set plus13 cold lepton-derivative/unit tests passes95 tests total in8.47s against the frozen candidate. The13 are a subset of95 and are not counted twice. This meaningful regression success does not erase the new counterexamples. No newer author98-test claim was relabelled independent execution. Environment uses the existing junior_tools interpreter/JAX x64 and current NumPy/SciPy; minimum/supported-version execution, full admitted repository CI, packaging and broad-domain fidelity remain open.

Every manifest-listed candidate file and the prior compatibility review SHAa3aa12992631504f83811ec569ac9fc044b91246f2cbe5325ed6733e94430f43 verify. The probe hashes the entire author tree before/after and finds no change. Source/manifest/report/checkpoint are pinned and copied to reviewed_candidate/. No fixes, installs, production checkout, branch/PR/git or paper edits occurred.

Commands: probe.py and diagnose.py run with JAX_ENABLE_X64=true,OPENBLAS_NUM_THREADS=1 and `/Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B`; original regression pytest command and raw output are recorded separately in this review directory. The pure pre-candidate oracle has no candidate imports.
