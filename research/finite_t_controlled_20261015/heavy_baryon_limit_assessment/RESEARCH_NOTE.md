# Artificial heavy-baryon limit of exact point-V-A capture moments

**Author-frozen forward benchmark; different independent review required.** The selected mass sequence approaches the same absolutely normalized stationary allowed EC/PC integrals without rescaling. All328 recorded checks pass:100 on-shell/current controls,62 mass-one forward checks,138 family numerical/provenance checks and28 preregistered limit diagnostics. This is an artificial limiting-family test, not a physical nucleon-mass prediction or an uncertainty estimate.

## Predeclared physical family

LOCK.json was written2026-10-16T03:05:59.909720Z before physical source calculations. Hold T3MeV,nB1e-7fm^-3,Ye=.5490065797957709, electron mass, net charge and incoming empty nue/antinue fixed. For lambda=1,2,4,8 define

`mbar0=(mn0+mp0)/2`, `Q0=mn0-mp0`,

`mn(lambda)=lambda*mbar0+Q0/2`, `mp(lambda)=lambda*mbar0-Q0/2`.

Here mn0=939.56542052MeV,mp0=938.27208816MeV,Q0=1.2933323600000222MeV. Q is mathematically fixed; its largest float64 representation difference is9.10e-13MeV at lambda8, below the predeclared2e-12 absolute gate. All actual deformed masses appear consistently in density inversions, spin projectors/current amplitudes, moving-target thresholds, CM energies, phase-space boundaries and outgoing mass shells. This is not a change of one weak-current denominator while keeping physical kinematics.

The literal current is **F1=1,GA=1.2723,F2=GP=0**, without q² dependence. The bare weak mass normalization is recorded as lambda*mbar0, but does not appear in this point current because its tensor/pseudoscalar terms are absent. It must not be interpreted as an RMF Dirac/Landau effective-mass variation. GF,Vud,hbar,HC, all spin/helicity conventions and the actual finite electron mass stay fixed. Relativistic moving-target FD distributions and final baryon/charged-lepton vacancies remain in the calculation; no classical baryon replacement is made.

Every lambda solves fresh particle-only free FD baryon roots at the same target densities, and the same neutral pair-electron root. The baryon chemical potentials therefore change with rest mass and fugacity; none is copied or shifted by a guessed Q correction. The electron EOS is identical across all profiles/scales. Baryon antiparticle diagnostics remain in each raw EOS; this is not a full pair-baryon EOS.

## Forward normalization before limit interpretation

The frozen EC and PC modules hardcode their physical mass constants. An isolated worker configures ALL relevant phase constants MN,MP,MW in both modules and explicitly passes those masses to the unchanged baryon density solver. Its scalar point-current function takes ordered initial/final masses as arguments. For EC the ordered inverse absorption pair is n->p; for PC it is p->n with the appropriate anti VA sign. Own source `point_current.py` contains only literal point V-A; there is no fitted factor.

The parent's orchestration imports no mutable physical modules. Every scale/profile runs in a fresh child, with deliberate child-only constant/amplitude configuration and complete override metadata. Upstream source bytes remain unchanged. Raw original module pin labels are interpreted together with this wrapper's deformed-mass/current provenance, not as an unmodified-physical-model certificate.

Before any lambda>1 run, mass-one scalar coarse/fine profiles AND an additional full-phase literal-spinor coarse calculation pass. The scalar-versus-spinor absolute number and energy moments agree within2.23e-16 for EC and PC. The latter uses the actual reciprocal spinors with the same point vertex and spin average4, not a rate normalization matched to the answer. PRE_DATA_ANCHOR.json records this extra forward calculation before data; no locked mass family, profile or gate was altered.

The100 separate current/kinematics controls use both ordered flavors at incident energies4.7/17.3MeV, cosines-.61/.22 and all four mass scales. Literal point form factors, direct-Dirac versus scalar amplitudes, positive amplitudes, deformed on-shell masses, four-momentum conservation and fixed Q pass. Inactive neutrino helicities annihilate through the chiral current; no incident-neutrino spin average or additional target factor is inserted.

Profiles remain EC20/32/8/8/8 tail50 ->28/48/12/12/12 tail60, PC28/44/10 ->32/52/12 tail60. Each moment separately passes3e-6 refinement; the largest family-profile difference is4.60e-10. FD target-density integrals pass3e-8, EOS n,p/net-charge roots1e-10, on-shell/four-momentum checks3e-10 and separate rest-inclusive species/direct energy ledgers3e-10. The largest reported family energy-ledger residual is4.29e-13. A shared quadrature implementation is not an independent physical model; the literal spinor anchor and analytic allowed limit serve distinct normalization purposes.

## Stationary allowed anchor and measured approach

The same-constant stationary per-target prefactor is

`GF² Vud² (1+3GA²)/(2*pi³*hbar)`.

It multiplies the finite-me relativistic electron/positron FD integrals with exact Q0 and actual fixed mu_e=.14119884195329963MeV. Target densities np=nB*Ye and nn=nB*(1-Ye) are fixed. These one-dimensional physical integrals are the independently accepted `weak_rates/static_capture.py` source, not a fitted asymptote. Their number/energy values equal the saved accepted allowed assessment at this state; no heavy-mass result enters their normalization.

Absolute anchor values are R_EC=4.849878328817105e-6fm^-3s^-1,Q_EC=7.070093372165171e-5MeVfm^-3s^-1,R_PC=5.530523135269570e-6fm^-3s^-1,Q_PC=8.813485608730491e-5MeVfm^-3s^-1.

The table gives100*(actual point-current finite-mass moment/allowed-1), preserving positive individual moments rather than a cancelled net:

| lambda | EC number % | EC nue energy % | PC number % | PC antinue energy % |
|---|---:|---:|---:|---:|
|1|-1.5603309|-2.2170406|-4.4379857|-5.5738214|
|2|-.7912759|-1.1280199|-2.2691188|-2.8633025|
|4|-.3984712|-.5690148|-1.1475045|-1.4515136|
|8|-.1999488|-.2857732|-.5770396|-.7308198|

All four absolute errors decrease strictly. Successive error ratios lie.50179–.51371, inside the preregistered[.25,.75] range and consistent with leading1/M corrections. All lambda8 moments pass the predeclared illustrative1% finite-M criterion. This selected four-point sequence supports the named limit; it does not establish a uniform convergence rate for all states or all masses. These finite-M differences are distances from an approximation, not integration error or physical uncertainty.

Mathematically, target velocities decrease as sqrt(T/M), recoil and relativistic spinor corrections disappear, and at fixed density the FD occupations dilute as M^-3/2. Thus the static unblocked-target point-current integral is the appropriate limiting reference. The finite-mass calculation retains all of those effects together. This study does not attribute the remaining difference uniquely to motion, recoil, Pauli blocking or relativistic V/A kinematics.

At lambda1 this is a different current from the previous full-current/F2-only assessment: ALL nonpoint form-factor/tensor/pseudoscalar effects are absent here. Its limit must not be presented as the physical full-current M->infinity limit with original q²-dependent form factors retained. The original allowed-model20 accuracy failures and the earlier F2-only results remain unchanged.

## Conservation and scope

EC is p+e- -> n+nue; PC is n+e+ -> p+antinue. The exact event measures include both initial charged/baryon spin states and one emitted active neutrino helicity. Rates are fm^-3s^-1 and neutrino energies MeVfm^-3s^-1. Rest-inclusive EOS and reaction energies use the same deformed masses; no separate Q or pair-rest source is added. Mass-family changes are not an evolution: no work/source term is assigned for moving between lambda values. No beta decay, complete weak inventory, net/root accuracy, cooling trajectory, radiation transport, interacting medium or derivative/AD inference is made.

Actual baryon total chemical potentials demonstrate the independent roots: at lambda1 mu_n=906.81094416,mu_p=906.11376298MeV; at lambda8 mu_n=7469.90311596,mu_p=7469.20053126MeV. Holding Q does not hold those chemical potentials or enforce beta equilibrium. Temperature/composition are prescribed throughout.

## Reproduction and preserved evidence

From Genesis, existing interpreter only:

```
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/heavy_baryon_limit_assessment/current_controls.py
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/heavy_baryon_limit_assessment/study.py baseline
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/heavy_baryon_limit_assessment/study.py limit
```

The baseline stage takes43.70s and limit stage11.73s, with9 combined physical calls total. Each child is externally bounded to110s and experiments remain below120s. Existing completed stage/child outputs refuse overwrite. Reproduction requires a new authorized sibling output directory with exact source/LOCK/anchor files; do not overwrite this frozen evidence. Full raw source moments, EOS/mass configuration, literal spinor anchor, partial checkpoints, stdout/stderr, numeric and limit gates and comparison.csv are retained.

All external physics/current/EOS/source/review hashes verify unchanged. No gate/profile was retuned after results and no failure is hidden. Runtime configuration occurs only in isolated children; no upstream bytes or parent physical-module globals change. No installs, dependencies, production checkout, paper or git/PR operation occurred. Different independent review is required before scientific promotion.
