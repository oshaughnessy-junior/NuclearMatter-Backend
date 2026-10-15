# Pinned O2scl matching fixture and independent equation reduction

**External matched finite-T GM1 forward gate remains OPEN.** No installed O2scl library was executed. This leaf improves the previous audit by pinning primary source, supplying a native fixture, and independently solving its reduced equations. Those numerical results are same-equation reimplementation checks, not an external-software oracle or physical error estimate. No frozen reference, table, parameter or production file was changed.

The earlier Q2C source audit identifies O2scl revision `d68532ff31e59dac65844c3d387e1388cb71a770`. The temporary checkout now contains residual Git metadata and no EOS sources. Primary files were therefore retrieved from the immutable GitHub URLs and saved under `sources/`, with original license notices and SHA256s in `results.json`: [RMF implementation](https://github.com/awsteiner/o2scl/blob/d68532ff31e59dac65844c3d387e1388cb71a770/src/eos/eos_had_rmf.cpp), [RMF definition](https://github.com/awsteiner/o2scl/blob/d68532ff31e59dac65844c3d387e1388cb71a770/src/eos/eos_had_rmf.h), and [relativistic pair-gas implementation](https://github.com/awsteiner/o2scl/blob/d68532ff31e59dac65844c3d387e1388cb71a770/src/part/fermion_rel.h). These are source snapshots for inspection, not a complete build tree.

## Exact matching convention

Target NMB revision `25e68182f726e5bd9c0b372c9a667536940fbd07`, common bare neutron/proton mass 939 MeV, HC=197.3269804 MeV fm, Cs=11.785, Cw=7.148, Cr=4.410 fm², b=.002947, c=−.001070. Uniform no-sea Hartree nuclear matter only: no leptons, photons, crust, nuclei or hyperons, no beta-equilibrium restriction. Both spin states for each baryon; thermal antiparticles retained, no zero-point Dirac-sea term. Net n_n=(1−Yp)nB and n_p=Yp nB. Total chemical potentials and energy include rest energy. Nothing is recalibrated to force agreement.

Use fm units in the solver, M=939/HC and t=T/HC. With Phi=gs sigma, W=gw omega and R=gr rho,

```
m*=M−Phi; W=Cw nB; R=Cr(np−nn)/2
mu_n=nu_n+W−R/2; mu_p=nu_p+W+R/2
U=Phi²/(2Cs)+b M Phi³/3+c Phi⁴/4
Phi/Cs+b M Phi²+c Phi³ = ns,n+ns,p
V=W²/(2Cw)+R²/(2Cr)
epsilon=epsilon_gas,n+epsilon_gas,p+U+V
P=Pgas,n+Pgas,p−U+V; f=epsilon−t s
```

The source at RMF lines116–188 implements precisely these signs after dividing its scalar field equation by gs. In particular it uses `b*ne.m`, so both particle masses AND `mnuc` must be set to the common M; importing physical neutron/proton splitting would create another model. The g_r tau3/2 convention gives Cr(np−nn)²/8 energy. Set `cs=sqrt(Cs)`, `cw=sqrt(Cw)`, `cr=sqrt(Cr)`, all additional couplings zero and `zm_mode=false`. The native fixture records meson masses550/783/770 MeV and gs=ms cs etc.; only their ratios enter this homogeneous model with no extra terms. This choice is not a finite-nucleus calibration. Set `inc_rest_mass=true` and interaction flags false for `non_interacting`. `pair_mu()` explicitly evaluates a particle and antiparticle and subtracts their densities while adding energy, pressure and entropy. This verifies the thermal-pair convention at source level.

## Executed reduction and evidence

`momentum_reduction.py` was written from the pinned equations. It integrates p directly with fixed Gauss–Legendre panels split at nominal kF, using measures p²dp/pi² (density), p²m*/E dp/pi² (scalar), p²E dp/pi² (energy), and p⁴/(3E)dp/pi² (pressure). Entropy uses the FD occupation entropy. It nests two bracketed density roots inside a scan for the first scalar upcrossing in Phi. It imports no internal state, root, quadrature, starting field or numerical value.

The frozen internal reference instead integrates kinetic energy with adaptive vector quadrature and solves three constraints together using bounded least squares initialized from its cold branch. Fresh internal forward states are compared only after both solvers independently reach their roots. This is useful numerical and convention independence, while the mean-field equations remain shared.

Two selected warm states `(nB fm^-3,T MeV,Yp)=(.16,10,.1),(.24,20,.3)` were tested. Orders96,160,256 at tail40T and order256 at tail60T are independently compared, including the tiny thermal antiparticle component. All **104 checks pass**, in **2.05 s** total including internal solves and failed native compile. Largest relative internal-forward difference is **1.08e−14**, and largest quadrature/cutoff difference **3.99e−14**, under the declared2e−9 component gate. Scalar/density residuals and the Euler identity are separate checks. No derivatives, phase coexistence, broad domain, or actual weak rates are qualified here.

| State | Phi [MeV] | mD [MeV] | mu_n / mu_p [MeV] | epsilon / P [MeV fm^-3] | s [fm^-3] |
|---|---:|---:|---:|---:|---:|
| .16,10,.1 |286.8415205464|652.1584794536|978.5262639335 /861.8227286774|151.7931270781 /4.23032933803|.1326510750914|
| .24,20,.3 |392.6480281907|546.3519718093|996.2633712382 /911.9010571275|227.4602303989 /11.45932302219|.2945215469920|

These table entries are **our independent reduction**, not published O2scl values. Antiparticle densities are respectively6.21535e−60 and3.85272e−27 fm^-3; no floor was used. At these states their negligible numerical magnitude does not justify removing them from the matching contract.

## Why external execution remains open

The available `/usr/bin/clang++` stops at `o2scl/eos_had_rmf.h: file not found`. The precise compiler output and checked library prefixes are in `results.json`. There is no local `o2scl-config` or candidate library in the checked Homebrew, /usr/local or existing conda prefixes. Docker is present; its initial sandbox socket access was denied, then read-only image inventory succeeded with reviewed escalation. No image was built, pulled or run. `docker_image_inventory.txt` records the installed images; it lacks the previously recorded Q2C native O2scl image digest `sha256:4725272cd7f860add7d4ba55e0f2e59bc8b305e4049eaf4ed447192155af8371`. Historical Linux `/usr/local/lib/libo2scl.so.0.22.0` metadata therefore do not attest a runnable library on this macOS host. This is a bounded availability audit, not proof no external runtime exists elsewhere.

`o2scl_fixture.cpp` is the minimal native matching request: independent prescribed densities/T, explicit parameter and unit settings, no internal converged fields supplied, O2scl's own density-based continuation, reported n/fields/mD/totalmu/epsilon/P/s/f and literal field residuals. It is **uncompiled/unvalidated as an executable** here. The exact inputs and expected internal-only references are in `fixture.json`. A future environment must build the complete pinned O2scl revision with its own dependencies, record source/library/compiler/solver controls, compile this driver, check its actual status and roots, and compare those outputs before derivatives. Do not relabel this reduction or the uncompiled fixture as external closure. Record native fermion/solver tolerances and tighten them independently before setting a numerical comparison gate.

The previous audit's cold beta-equilibrium CompOSE/Zenodo GM1 tables remain incompatible with this finite-T independent-Yp nuclear-only target. We did not import another GM1 calibration, infer a thermal extension from cold data, or fit couplings to saturation values. External physical validation and global-phase selection remain separate open gates.

Reproduce the bounded executed checks with `/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/external_gm1_forward/check_forward.py`. The native compile failure is retained as an availability result, not counted among numerical checks. Source snapshots and hashes make a future external invocation reviewable; no production promotion is proposed.
