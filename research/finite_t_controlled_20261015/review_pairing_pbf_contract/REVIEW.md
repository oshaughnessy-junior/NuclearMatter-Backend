# Independent PBF phase/source contract review

**ACCEPTED only for the selected finite-temperature legacy parity and reduced-integral scope, with two newly failed subnormal-input extensions explicitly excluded.** Independent results:92 PASS,2 FAIL. The four documented raw legacy endpoint failures remain excluded and are freshly reproduced. No scientific author source was edited. Reviewed helper `quasiparticle.py` SHA256d55ee3540766d00971f036245cb8bf0950114526a6230e7f7e7c28c44a0dc1b4; author MANIFEST6e54face569daaac2be62681011dd229f910681843dd25899cc48d22c77aea05. The parent dispatch's `phase.py` name refers to this actual `quasiparticle.py` file.

## Independent forward construction

I changed the radial variable to quasiparticle energy z, with z=v+t². Since dx/dz=z/sqrt(z²−v²), the exact singlet function is

    FA(v)=2v² exp(−2v) integral_0^infinity dt
          (v+t²)^5/sqrt(2v+t²) * exp(−2t²)/(1+exp[−v−t²])².

The endpoint Jacobian is finite. The implementation evaluates the positive rescaled integral and adds2log(v)−2v; it never forms v² at tiny gap. The radial upper endpoint t=9 has exp(−162) suppression after extracting exp(−2v); the selected comparisons provide numerical evidence rather than a uniform analytic remainder certificate.

For the full B angular gap, I changed from polar cosine to rapidity: mu=sinh(theta)/sqrt3, u=v cosh(theta), theta∈[0,asinh(sqrt3)], dmu=cosh(theta)dtheta/sqrt3. A separate symmetric sphere integral in the rms convention checks the4pi normalization and Delta_rms=sqrt2 Delta_min identity. This is different from the author's scaled-x and adaptive-mu reduction, and different from its x=v sinh(t) alternate radial rule. All methods still evaluate the same declared reduced BCS model; this is not independent many-body physics.

Ten new heldout gaps1e−250,1e−163,.017,.43,2.7,9.3,37,127,300,499 pass FA/FB comparisons at3e−10 relative. Maximum log differences are 0 and 7.11e-15. Small-gap series coefficient24[eta(5)−eta(4)] and angular factor2 are independently recovered. Large-v Laplace limits at350/500 and the three rms/sphere checks pass. Fourteen fresh ordinary-valued legacy-fit comparisons at v=.017,.43,2.7,9.3,37,127,300 pass the original5% gate; maximum error is 1.210%. This is selected approximation error to a reduced scalar function, not a physical PBF uncertainty.

## Fresh pinned-source execution and absolute units

I independently verified the Git blob identities of both NMB snapshots, inspected the literal function/constant AST extraction, and extracted the literal local Fortran subroutines into my own leaf. A fresh GNUFortranO0 driver with ca varied separately executes new tau=.231,.672,.973 andabove-Tc1.07 states at varied Tc,mst,kF; all8 finite/above-Tc channel comparisons pass3e−12. Both float32 literal widening and full source-prefactor conventions remain exact. This demonstrates parity with the pinned **local patched** NSCool source, not fresh official release execution or independent physical validation.

The Landau effective mass convention mL=pF/vF is essential. I independently converted GF from GeV−2 and kB/hbar from SI, giving cgs coefficient1.17098013112e+21 rather than rounded1.170e21. The source-rounded difference passes. [Kaminker, Haensel and Yakovlev1999 Eqs1–2,6–7](https://arxiv.org/pdf/astro-ph/9904166) support the phase function, T^7, three-flavor energy normalization and stated singlet axial coefficient. Omitting the old vector contribution is a separately named response approximation; phase-integral parity does not validate a conserving weak tensor. kF is fm−1, local temperatures are kelvin and output is energy emissivity erg cm−3 s−1, not a number rate or opacity.

[Leinson2010 Eqs96–99](https://arxiv.org/html/0912.2164v2) confirm the replacement of CV²+2CA² by CA²/2 in the stated BCS/no-residual-particle-hole approximation. With CA1.26 the matched-gap ratio is 0.190122628856; relative to vector-omitted2CA² the ratio is1/4. These require the same physical angular gap. That paper explicitly distinguishes rms and minimum amplitudes bysqrt2. It does not authorize a universal rescale across different gap prescriptions or provide a modern physical uncertainty estimate.

[Yakovlev, Levenfish and Shibanov1999 Eq11](https://arxiv.org/pdf/astro-ph/9906456) gives the triplet minimum-gap1.188/tau coefficient. The source's1.764/tau differs and is not an exact rms/minimum relabelling. Its temperature-dependent PBF changes therefore are prescription changes, not a few-percent fit residual. PBF's i<=isf triplet boundary is verified at newzones14/15/16; its difference from Urca's i<isf convention remains explicitly preserved.

## New failed numerical boundary extension

`extreme_results.json` preserves both independent failures unchanged. At v=1e−320, logFB differs from2log(v)+log(2*small-gap coefficient) by relative3.917239779e−6 at the unchanged3e−10 gate and emits a QUADPACK maximum-subdivision warning. At v=nextafter(0,+1)=4.940656458e−324, the error is3.175416347%. Multiplying this subnormal input bysqrt(1+3mu²) rounds the physical gap argument into a few discrete floating values before evaluatinglogFA. A log-domain radial normalization cannot recover angular information already rounded in that multiplication. I did not increase quadrature subdivision limits or relax the gate to hide it.

This is an **input** representability limitation. Explicitly exclude0<v<sys.float_info.min=2.2250738585072014e−308 from any accuracy-qualified helper admission unless separately repaired. The current API still accepts those arguments, so broad accuracy over its literal0<v<=500 admission would be false. The author selected1e−200 and my fresh1e−250,1e−280,1e−307 points pass; this does not prove a uniform rectangle. Parent will handle any admission repair separately after this frozen review.

Ordinary **output** underflow is different and remains allowed: at the smallest positive NORMAL input, independent angular coefficient error is−5.50e−14, logFA=−1417.3001910833 andlogFB=−1416.6070439027 remain finite, while both ordinary positive functions correctly carryNone/unrepresented flags. Deep-large-gap inputs similarly retain finite logs. No floor or artificial exact-zero output is introduced. The old v=1e−200 helper failure is independently replayed against its exact preserved initial code, and repaired normal-input results pass.

## Preserved endpoints and accepted limits

The fresh raw T0 calls return NaN in both NMB channels. At exactTc, fresh localFortran returns zero while literal NMB returns small positive clamp artifacts in both channels. These four failures remain failures; they are not included in the92 passing checks. Wrapper rejection of zero temperature and tau<.01 does not repair those scientific sources. The zero-gap v=0 phase limit and zero-temperature v→infinity limit are distinct limits even though the emissivity vanishes in both.

This acceptance qualifies selected legacy function execution, source-labelled gap/current conventions and the reduced FA/FB reference. It excludes subnormal input accuracy, full conserving/anomalous weak-current tensors, paired EOS heat capacity/entropy/condensation energy, gap derivatives/AD, collective spin-wave emission, correlated or realistic matter uncertainty, opacity and a complete cooling trajectory. A PBF sink added to an unpaired EOS does not establish thermodynamic consistency.

## Frozen evidence

`probe.py` produces73 PASS in2.89s; `fit_probe.py`14 PASS; `extreme_probe.py`3 PASS and2 retained FAIL; the independently executed smallest-normal boundary probe adds2 PASS. `results.json`, `fit_results.json`, `extreme_results.json` and `boundary_results.json` retain values, controls and failures. All author bytes are unchanged and the exact full manifested initial author tree is copied to `reviewed_initial_author/` before any parent-authorized repair. `CHECKPOINT.json` records this review's hashes. No author driver was imported or executed and no dependency/production/paper/git state was changed.
