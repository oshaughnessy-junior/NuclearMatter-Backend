# PBF pairing, phase-integral and weak-current contract

**This leaf executes the pinned cooling prescriptions and independently normalizes their reduced quasiparticle phase functions. It does not qualify a complete conserving pairing response, paired EOS or cooling profile.** There are132 selected passing controls and four explicitly retained raw endpoint exact-zero failures/exclusions. Scientific NMB/NSCool files were not changed. A different independent reviewer is required.

## Actual source and existing cooling scope

`source_snapshot/` preserves exact NMB commit25e68182f726e5bd9c0b372c9a667536940fbd07 files: core_nu.py Git blob520d3b458c5eddbed1e1e1a5c6270dcc12643eb5 and superfluid.py blobe16a07cfa7b92530649822358c552b1fa7069042. `contract.py` checks those immutable blob identities and executes the exact FunctionDef and required constant-assignment ASTs in real JAX64. Function bodies, float32 Fortran-literal widening and legacy constants remain literal. Production imports and unrelated tables/drivers are omitted.

The earlier C2 cooling report records zone-parity errors of2.2e−6 for neutron1S0,2.7e−6 for neutron3P2 and4.5e−6 for proton1S0 in its selected historical stellar configurations. Those are source-prescription/profile qualifications, not modern weak-current accuracy. The staged physics README already records the PBF/Urca zone and triplet-gap discrepancies. Their exact read-only hashes are in `SOURCES.json`; this leaf neither reruns nor expands those cooling qualifications.

`fortran_probe.py` independently compiles the literal PBF subroutines extracted from the pinned **local installation** source SHA45fb3891d0d2c69a527b0fa64be82eb69fd261f6ccccbc01b4d706ef4e7d9def. That installation contains the documented nmbackend ca patch and is not asserted to be a fresh official release. Eight fresh nonendpoint/above-Tc Fortran/actual-NMB comparisons pass at3e−12; actual differences are machine precision. ExactTc disparity is separately retained. This tiny compile uses existing GNU Fortran and is not a whole NSCool build or a new independent physical model.

## State, gaps, current weights and channels

T and Tc are **local kelvin**, kF is fm^-1, and mst is the **Landau/bare mass ratio**. These are not Dirac masses, net densities or MeV temperatures. The scalar output is local total neutrino energy emissivity in erg cm^-3 s^-1. It sums all three flavor pairs, with Nnu=3; it does not return number rates, flavor spectra or an absorption opacity. Each pair has one neutrino plus its antineutrino, zero net flavor lepton number and no material proton/neutron conversion. No extra beta-decay Q term belongs here.

The literal scalar prefactor is

```
vF/c = (197*kF)/(940*mst)
Q = 1.170e21 * mst² * (vF/c) * (T/1e9K)^7 * 3 * a * F(v).
singlet a = 1.60*(vF/c)²*(mst²+11/42), vector omitted;
triplet neutron a = ca*1.26², default ca=.5, vector omitted.
```

`ca` is a **multiplier of the squared axial response coefficient**, not the weak axial vertex coupling used by other scattering modules. Do not insert.5 as an independent NC vertex or multiply a second suppression factor onto the already modified source. The singlet expression is used by the legacy prescription for both neutron and proton1S0; neutron3P2B is the only triplet channel here. Approximate197MeV fm and940MeV conversion constants are retained.

`normalization_anchor.py` converts the reduced expression `4GF² mL pF T^7/(15pi^5) * Nnu*a*F` from natural MeV^5 to cgs using `MeVerg*1e39/(HC³*hbar)`. With Mref940MeV it gives1.17098013e21, differing from the source's rounded1.170e21 by.08377%. This is an independent units/prefactor anchor, not a new calibration or weak-current validation. The pairing quasiparticle sum has its own identical-particle counting; importing ordinary free-target spin averages again would double count.

For tau=T/Tc and v=Delta/T, the named gap variants are

```
YK1999_A: sqrt(1−tau)*(1.456−.157/sqrt(tau)+1.764/tau)
YK1999_B_minimum: sqrt(1−tau)*(.7893+1.188/tau)
NSCool_NMB_B_legacy: sqrt(1−tau)*(.7893+1.764/tau).
```

[Yakovlev, Levenfish & Shibanov1999 Eq11](https://arxiv.org/pdf/astro-ph/9906456) supplies the first two. The third is the actual PBF source, distinct from its Urca helper. B denotes the nodeless mJ=0 gap, `Delta(theta)=Delta_min*sqrt(1+3cos²theta)`. A normalized rms-gap convention instead has `Delta_rms=sqrt(2)*Delta_min` and `bbar²=(1+3cos²theta)/2`. The literal legacy1.764 coefficient is not silently relabelled as that exact conversion.

PBF dispatch assigns i<=isf to neutron3P2 and i>isf to neutron1S0. Urca assigns i<isf to3P2 and i>=isf to1S0. Thus the boundary zone intentionally differs in the preserved prescription; this is not a universal physical rule or a newly repaired inconsistency. Independent per-zone gap choices are needed for a new model.

## Reduced integral: what is independently tested

[Kaminker, Haensel & Yakovlev1999 Eqs1–2](https://arxiv.org/pdf/astro-ph/9904166) give the normalized singlet phase function and the relativistic proton axial correction. The bare vector term in that earlier treatment is separate; retaining its phase function does not retain that vertex prescription. Our executable dimensionless reference is

```
FA(v)=v² integral_0^infinity dx z^4/(1+exp(z))², z=sqrt(x²+v²).
FB(vmin)=integral_0^1 dmu FA(vmin*sqrt(1+3mu²)).
```

FB retains the full stated angular gap, rather than replacing it by an averaged gap inside the quasiparticle energy. The accepted literature fits in actual NMB `_ctrl_1s0/_ctrl_3p2b` approximate these scalar functions. They do not independently implement an angular weak-current tensor or collective modes.

`quasiparticle.py` computes logarithms using stable exp[−2(z−v)] scaling; `z−v=x²/(z+v)` avoids cancellation. The angular integral is normalized by logFA(v), preserving positive values below ordinary float64 representation at large OR tiny gap. `check_alternate_integral.py` changes to x=v sinh(t), z=v cosh(t), uses explicit energy tails50/70 and fixed64/96 angular rules. It agrees with the scaled-x/adaptive-angle reference within2.85e−14 on four distinct v values. Numerical method agreement concerns the stated integral only.

On v=.001,.03,.2,.7,1.5,3,6,12,25,60,100, the maximum legacy-fit relative errors are **1.22043% for FA and3.21763% for FB**, within a predeclared5% selected comparison gate. They are numerical approximation errors to the reduced integral, not total PBF physical uncertainty. The independent small-v coefficient is `24[eta(5)−eta(4)]=.602086582792`; FB has twice it. Low-T checks approach `FA~sqrt(pi)/2*v^(13/2)*exp(−2v)` and `FB~pi/(4sqrt(3))*v^6*exp(−2v)`. The leading asymptote's finite-v remainder decreases as v grows; it is not forced to vanish at moderate v.

Positive logFA(1000)=−1955.21918447 and logFB(500)=−963.497209008 remain finite while their ordinary values are explicitly **None/unrepresented**, with no floor. One initial new-helper failure at v=1e−200 was an angular rescaling underflow producing a math-domain error. Exact failed code and input/result are preserved as `initial_small_gap_*`; baseline logFA normalization repairs this research helper, and tiny-gap analytic regressions pass. No NMB source or fit gate was altered.

## Distinct weak-current prescriptions

[Leinson & Pérez2006](https://arxiv.org/pdf/astro-ph/0606651) show why the old bare vector response is not a conserving current calculation. Omitting the leading vector term is an approximation; it is not a finite-q Ward-identity proof or a complete treatment of charged-condensate plasma effects. The source retains the historical singlet axial term; no universal new singlet suppression factor is invented here.

For triplet B, [Leinson2010 Eqs96–97](https://arxiv.org/pdf/0912.2164) explicitly replaces the bare `(CV²+2CA²)` weight by `CA²/2` in its BCS/anomalous-vertex treatment. With CV1, CA1.26 and the **same physical angular gap**, the coefficient ratio is.190122629. Relative to vector-omitted bare axial weight2CA² alone it is1/4. This sources the legacy default ca=.5 comparison, while the paper's response approximations and neglected Fermi-liquid effects remain material qualifications. We have not implemented its full tensor or collective spin-wave radiation.

Gap changes must be separated from that matched-gap current comparison. Independent FB evaluated with legacy versus YK1999 minimum-gap amplitudes gives ratios1.70231e−9 at tau.05,.0402272 at.2,.785799 at.5 and1.78768 at.9. These large, temperature-dependent prescription effects are not a numerical fit residual, an uncertainty band or proof that one named cooling model has been repaired.

## Tc and zero-temperature exclusions

`below(...,strict=False)` admits T<=Tc but clamps tau to at most1−1e−12. In the selected pre-clamp ladder the rates approach their linear Tc slopes; six controls pass. At exactTc the literal NMB calls instead return small positive values (for mst.78,kF1.9,Tc1e9:2.23440e9 singlet and6.79243e9 triplet cgs). Original local Fortran returns exactly zero. These are **failed literal exact-zero endpoint gates**, retained as a named guard artifact, not relative errors against a zero denominator. The clamp saturation is explicit below1e−12 distance from Tc. AboveTc both return zero.

At T=0,Tc>0, the raw NMB functions return NaN from infinite gap/T algebra. Both raw zero-T finite/zero gates remain excluded. The research wrapper rejects T=0 and tau<.01 rather than replacing those results or asserting whole-domain cold stability; the normalized integral supplies a controlled low-T/log reference separately. Wrapper admissions are scalar numerical controls, not a certificate of physical degeneracy or NR validity. A caller must independently establish strong degeneracy, weak-coupling gap scales and the relevant matter/current assumptions.

## Evidence and integration recommendation

Counts:101 main source/phase/branch controls,16 alternate-coordinate checks,6 pre-clamp Tc checks,8 local-Fortran comparisons and1 independent cgs anchor, **132 PASS**. Four raw endpoint exact-zero gates remain false and excluded; the new-helper tiny-gap failure is separately preserved. Each experiment is small; the main numerical suite runs below1s and no install or full backend build was used. Commands and hashes are frozen with the raw JSON files.

Keep future NMB interfaces separate: a named gap convention and Tc(n,composition) provider; a dimensionless phase-function representation with measured fit errors; a named weak-response/current prescription with matching gap normalization; and an energy-emissivity adapter with explicit flavor/units. Existing legacy cooling remains an available parity mode. A self-heating PBF configuration additionally needs paired EOS heat capacity, entropy/condensation-energy consistency and gap derivatives; attaching this sink to an unpaired EOS does not establish that closure. No new full cooling trajectory, pairing EOS, realistic interaction uncertainty, radiation opacity or AD qualification is claimed here.

Later independent review found two subnormal-input extensions fail; the original92PASS/2FAIL/4 known-endpoint-FAIL record remains frozen. `ADMISSION_REPAIR.md` documents a subsequent scoped admission repair: both phase-log helpers accept exact zero or normal positive float64 inputs only, preserving ordinary output-underflow/log handling and every scientific formula/tolerance. Exact original artifacts and reviewer failures are retained; separate repair recheck is required.
