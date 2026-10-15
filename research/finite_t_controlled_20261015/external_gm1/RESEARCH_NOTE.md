# External finite-T GM1 forward benchmark: candidate audit

15 October 2026. This new research leaf audits external primary sources against the campaign's exact rounded NMB GM1 convention. No frozen research or production implementation was changed. No package was installed, external EOS library executed, large table downloaded, derivative compared, or parameter refitted.

**Outcome: the external finite-temperature forward gate remains OPEN.** O2scl provides a strong configurable independent-software candidate, with a concrete matching recipe below. The inspected public GM1 tables do not supply the requested finite-T, independently specified-composition nuclear state. An external runtime was not available in the checked existing environment. This is a bounded negative audit of those candidates, not a claim that no suitable implementation or dataset exists anywhere.

## Exact target

The target is the existing finite-T reference at immutable NMB snapshot `25e68182f726e5bd9c0b372c9a667536940fbd07`: common bare mass **939 MeV**, `Cs=11.785`, `Co=7.148`, `Cr=4.410 fm²`, `b=.002947`, `c=−.001070`, zero additional vector/cross couplings. It is uniform nucleonic no-sea Hartree matter at independent net `(nB,T,Yp)`, with thermal nucleon/antinucleon occupations retained. There are no leptons, photons, nuclei, hyperons, neutrinos or beta-equilibrium constraint in this nuclear-only target. Total energies/chemical potentials include the common rest energy; no vacuum Dirac-sea term is added.

The exact match is to these rounded couplings, not to another calibration sharing the label GM1 or nominal saturation properties. A cold beta-equilibrium npe curve cannot determine the target finite-T state function at independent Yp. `comparison_contract.json` records the target, explicit candidate conversion, internal forward payload hash and five **clearly labeled internal-only** sample states. Its `external_results` is null.

## Primary-source candidate decisions

| Candidate | What was actually verified | Exact match / mismatch | Decision |
|---|---|---|---|
| [O2scl `eos_had_rmf`](https://awsteiner.org/code/o2scl/html/class/eos_had_rmf.html) | Independent-density finite-T RMF interface and configurable coupling/field conventions | Configurable to the reduced target; execution and branch comparison still needed | **Preferred execution candidate**, not yet an external numerical benchmark |
| [O2scl implementation](https://github.com/awsteiner/o2scl/blob/main/src/eos/eos_had_rmf.cpp) | Finite-T path calls particle/antiparticle gas evaluator; scalar/vector equations and energy/pressure assembled separately | Source structure supports matching; a pinned revision/build and exact flags remain required | Audit source before numerical execution; do not treat moving `main` as a frozen result |
| [CompOSE GM(GM1), EOS 54](https://compose.obspm.fr/eos/54) | T=0 only, beta-equilibrated npe, with an attached crust | Fails finite-T and independent-composition requirements; includes electrons and nonuniform low-density treatment | Reject as finite-T forward gate; possible later matched cold-core test |
| [Zenodo 14803085](https://zenodo.org/records/14803085) | Archive explicitly identifies its GM1 dataset with CompOSE 54 | Same cold dataset, not a second independent finite-T source | Do not count as an additional finite-T oracle |
| [CompOSE XMLSLZ(GM1), EOS 282](https://compose.obspm.fr/eos/282) | Unified cold beta-equilibrium npe-muon matter with Thomas–Fermi/Wigner–Seitz treatment | T=0, composition/phase/species differ | Reject as finite-T independent-Yp gate |
| [Tsiopelas, Sedrakian & Oertel finite-T hypernuclear tables](https://link.springer.com/article/10.1140/epja/s10050-024-01351-1) | Public finite-T 3D CDF program includes baryon octet and chosen symmetry-energy slopes 30/50/70 MeV | Different interaction calibration/species and low-density matching; not the target GM1 | Useful broader comparison family, not an exact GM1 validation |

No cold table was converted into a finite-T result using a fitted thermal-index extension. No hyperonic or differently calibrated table was silently relabeled nuclear-only GM1. Published saturation-density/energy values are useful convention checks but do not close the finite-T forward gate.

## O2scl exact-configuration audit

The following matching recipe is derived from the target equations and the independent source's documented variables. It is a **proposed configuration**, not a record of executed software. O2scl's class documentation and [particle-unit documentation](https://awsteiner.org/code/o2scl/html/part.html) distinguish its raw variables from the target.

| Convention | Target / required external setting | Audit status |
|---|---|---|
| Input state | `ne.n=(1−Yp)nB`, `pr.n=Yp*nB`, `T_external=T_MeV/197.3269804` | Density-based finite-T entrypoint exists |
| Bare masses/spin | Both fermions `m=939/HBARC`, degeneracy 2; set nonlinear scalar mass scale consistently | Must override physical separate n/p masses/defaults |
| Scalar/vector ratios | `cs=sqrt(11.785) fm`, `cw=sqrt(7.148) fm`, `cr=sqrt(4.410) fm` | **Important square-root conversion**; external lowercase c has units fm |
| Scalar nonlinear terms | b=.002947, c=−.001070, with `U=Phi²/(2Cs)+b M Phi³/3+c Phi⁴/4` | Same polynomial convention; do not confuse kappa/lambda conventions |
| Isovector vertex | g_rho*tau3/2; `R=Cr(np−nn)/2`; energy `Cr(np−nn)²/8` | Half-vertex convention matches |
| Additional interactions | zeta=xi=a1..a6=b1..b3=0; `zm_mode=false` | Required explicit zeros, no density-dependent couplings |
| Rest-energy output | Both fermions `inc_rest_mass=true`; interacting flags appropriate to RMF | Avoid hidden rest subtraction in e,mu,nu |
| Thermal occupations | Both particles and antiparticles; net target densities | Finite-T source calls `pair_mu` for each nucleon |
| Vacuum | Omit zero-point/Dirac-sea energy/counterterms | Inspected finite-T expression contains gas+mean-field terms; verify pinned build |
| Species | Invoke nucleonic RMF class alone | Do not add a beta solver, leptons, photons, hyperons or crust |
| Fields for comparison | Convert g_s*sigma, g_w*omega, g_r*rho and m* to MeV | Compare rescaled physical fields, not differently normalized raw meson fields |
| Phase | Homogeneous locally continued branch | No automatic Maxwell/Gibbs/nonuniform replacement |
| Units on return | e,P,f [fm^-4]×HBARC; mu,m*,fields [fm^-1]×HBARC; s [fm^-3] | Use the exact declared HBARC, record library constants separately |

Source-level detail matters: the inspected finite-T scalar self term uses the neutron bare-mass scale in its coefficient. Setting **both** particle masses and the class scale to 939/HBARC avoids a documentation-versus-implementation scale ambiguity. This is not a reason to refit b.

Individual positive meson masses must be explicitly recorded in an eventual harness. In uniform matter with these extra couplings zero, the reduced equations depend on the fixed ratios g/m; choosing a different individual meson mass at the same ratio is not a saturation refit of this homogeneous model. Finite-size/gradient comparisons would need the actual individual masses and are outside this contract.

## Why no external forward numbers were generated in this batch

`check_availability.py` was executed with `/Users/rossma/miniconda3/envs/junior_tools/bin/python`. It found no `o2sclpy` module, no `o2scl-config` on PATH, and no filename containing o2scl in the checked `/opt/homebrew/lib`, `/usr/local/lib` and existing junior_tools library directory. This is a bounded availability check, not a whole-computer inventory. The machine's available Python/NumPy/SciPy cannot by itself execute the independent O2scl C++ EOS.

The [official installation documentation](https://awsteiner.org/code/o2scl/html/install.html) requires an O2scl build and dependencies including Boost, GSL and HDF5. This batch explicitly forbids installs/heavy builds, so it did not create a different implementation and call that external, extract a few equations into the campaign's own solver, or install a wrapper with no linked library. An existing compatible external runtime or externally produced state payload is the concrete missing input.

The browsed O2scl source is public but the inspected URL is moving `main`; the commit-history/API fetch did not yield an immutable revision through the browsing tool. Pin and hash the actually executed revision before any accepted comparison. Some raw/datasheet URLs returned cache/access failures; the audit relies on the accessible primary catalogue, docs and rendered implementation, not unread bytes. No external numerical result is inferred from a fetch attempt.

## Minimal executable follow-up once a runtime exists

1. Pin the independent source revision, linked library build and constants. Assign every setting above explicitly rather than loading an unverified GM1 label. Record the actual meson masses and flags. Retain a minimal standalone external harness, its stdout and source hashes separately from the campaign oracle.
2. Start with the three warm stable-coordinate examples `(nB,T,Yp)=(.16,10,.1),(.24,20,.3),(.32,30,.1)`. Solve at fixed species densities with external cold-to-warm continuation; inspect residuals and branch. Do not use agreement of two numerical roots to infer global phase stability.
3. Serialize **forward** scalar/vector/isovector fields, mD, kinetic and total chemical potentials, e,P,s and f=e−Ts. Include independent species number residuals. Compare all values before any derivative test.
4. Tighten external quadrature and root tolerances and perturb starting guesses within the same branch. Assess differences relative to measured numerical errors and declared constant rounding. No discrepancy is repaired by altering Cs/Co/Cr/b/c, rest masses, or an energy offset without documenting that the original comparison failed.
5. After matched forward agreement, compare fixed-composition thermal derivatives and parameter sensitivities. An external forward test at a few points still does not establish all-state physical accuracy, a phase construction, a paired EOS or consistent weak-current vertices.

The existing internal sample values in `comparison_contract.json` are ready as one side of that comparison and are explicitly not evidence that the missing external side has run.
