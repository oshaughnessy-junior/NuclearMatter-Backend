# Total physical positron capture/inverse thermal event moments

Research prototype for ONLY n+e+ <-> p+anti-nu_e, in a fixed homogeneous, isotropic physical-mass free npe FD bath. It retains actual initial/final FD factors, exact relativistic two-body phase space and the frozen full reciprocal current. It is not an EOS replacement, a cooling trajectory, or a complete antineutrino inventory. In particular, older low-T neutron-beta moments are not added to this T2 bath. Independent review is pending.

## State, current and normalization

The selected neutral pair-rich bath is T=2 MeV, nB=1e-7 fm^-3, Yp=Ye=.3, with muons absent. Actual FD roots give mu_n=919.8287578100662, mu_p=916.8449303084382, mu_e=.17446333200826397 MeV; mu_e+=-mu_e. Neutrality uses net electron density, not the electron particle density alone; positron/electron density ratio is .8525281373. MATCHED_STATE.json records all densities, energy and anti-baryon diagnostics. Nucleon masses are mn939.56542052 and mp938.27208816 MeV; electron mass .51099895069 MeV. Chemical potentials and event energies include rest masses. Matter has no photons, muons, interactions or thermal antiparticle baryons.

The current is the accepted [Guo et al. massive-current model](https://arxiv.org/html/2006.12051v2), Eqs26–29 and Appendix B, as implemented by independent explicit-spinor primitives in relativistic_capture_oracle/events.py. Vector/axial, F2 and GP terms and physical q² form factors are retained with bare weak mass (mn+mp)/2. For the reciprocal amplitude anti+p -> n+e+, Gamma_pn(q)=Dirac-adjoint Gamma_np(-q). The same invariant squared amplitude is used for the reversed event, with actual occupations reversed; it is not a capture tensor evaluated at an unchanged crossed transfer.

For initial neutron along z and positron at relative cosine c_i, define s=(pn+pe)². Initial/final CM magnitudes are pi=lambda(s,mn²,me²)^(1/2)/(2sqrt(s)), pf=(s-mp²)/(2sqrt(s)). The final antineutrino CM direction has cosine c_o to the bath boost and azimuth phi. Exact boosts return final proton and antineutrino energies to the bath frame. Neither an inserted physical width nor a stationary-target energy cut is used.

    v_M = pi sqrt(s)/(En Ee)
    dσ/dΩ_CM = |M|²_avg4 /(64π²s) pf/pi
    dR = [pn En dKn/π²] [pe Ee dEe dc_i/(2π²)]
         × dcos_o dphi × v_M dσ/dΩ_CM /(hbar hbarc³) × occupations.

Both incoming measures include spin2. The frozen antineutrino-capture spinor primitive averages initial proton spin2; for n+e+ the initial spin average is4, so its squared amplitude is divided by2. This is paired with the two spin2 measures, not an additional neutrino degeneracy. Number rates have units fm^-3 s^-1; energy rates MeV fm^-3 s^-1. In natural units the measures, velocity and cross section give MeV4, divided by hbar gives MeV3/s; hbarc^-3 converts the volume. No neutrino DOS is inserted into an already complete event integral.

## Direct forward/backward products, radiation and ledgers

For common-T FD radiation with prescribed mu_anti, independently integrated products are

    F = fn fe+ (1-fp)(1-fanti),
    B = (1-fn)(1-fe+) fp fanti.

Empty radiation sets fanti=0, making B exactly absent. It does not remove material blocking. Positive F and B number/energy moments are accumulated separately in log space. Net moments use an expm1-stable pointwise difference, preserving cancellation near LTE. No KMS relation constructs B. It is checked afterwards as log(F/B)=(mu_n-mu_p-mu_e-mu_anti)/T. Empty, LTE and mu_anti=mu_eq-.4T scenarios are evaluated using the same physical events. This radiation closure is prescribed and isotropic; there is no radiation evolution.

Each signed forward event gives dn=-1, dp=+1, dpositron=-1, d(nelectron,net)=+1 and danti=+1. Material Le increases by1 and radiation Le decreases by1. Rest-inclusive material energy changes by Ep-En-Ee=-Eanti. The code accumulates neutron, proton, positron and antineutrino energy moments separately, plus direct material-loss En+Ee-Ep; their independent summation ledger is checked. There is no added Q source. For common-T FD radiation total entropy production is integral (F-B)log(F/B) >=0. Algebraic conservation/positivity does not establish quadrature accuracy or a coupled-time solution.

## Numerical design and independent anchors

Incoming kinetic energies use exact square maps K=T y² and ordinary Gaussian quadrature, not a classical FD replacement. All occupations remain FD at every point. Baryon and positron upper bounds include thermal tails and final-baryon/prescribed-radiation chemical centers. Finite empirical tails are explicitly refined rather than claimed as analytic uniform bounds. API guards restrict the prototype to dilute baryons max(eta_n,eta_p)<=-5, pair-rich |mu_e/T|<=3 and T=.2–4 MeV; these are evaluation guards, not an accuracy certificate over that rectangle. FD radiation also requires |mu_anti/T|<=10. Extreme cold/degenerate or arbitrary chemical-affinity tails require a new integration design.

Three independent vacuum source-polynomial cross-section anchors use inverse anti+p kinematics at the SAME invariant s. Exact detailed cross-section balance gives σ(n+e+ -> p+anti)=(1/2)(pf/pi)² σ(anti+p -> n+e+). The factor1/2 is the initial-spin degeneracy ratio. Explicit-spinor forward CM integration matches these absolute source-polynomial predictions at three incident positron energies, without a normalization fit.

A separately ordered conditional spectrum solves the boosted two-body cone at fixed Eanti, integrating its delta function over CM cosine. It uses the exact Jacobian 1/(gamma beta Eanti_star) and root-partitions positron-energy support before integration. The resulting dR/dEanti divided by one-helicity DOS Eanti²/(2π² hbarc³) gives j and chi for comparison with the accepted response pilot. This is an additional absolute normalization and selected-spectrum check, not a full response-integrated neutrino-energy benchmark. The conditional method shares spinor/current physics but has different quadrature support ordering.

## Preserved failures and scope

The first coarse study remains unchanged in results.json: 62 failed checks among102, including actual neutron-density failures at radial8/12 and moment differences above the unchanged3e-6 gate. Radial16's good density alone does not qualify its moment accuracy. Later selected qualification is in refinement.json, while controls_anchor.json records malformed-input and independent vacuum gates. Do not combine the coarse pass/fail count into an all-pass claim.

The finite-T full-current model omits Coulomb/radiative/medium effects, mean fields, nucleon correlations and beta decay. Method agreement measures numerical differences inside the stated model, not physical uncertainty. A complete common-bath antineutrino source requires separately qualified beta/inverse3body moments at THIS bath before adding channels. No full-inventory, all-energy, broad-domain, material evolution or derivative claim follows here.

## Completed selected results

Four qualified completed total runs use n/e orders20/40 and28/56 with initial/outgoing/phi8, then angular12 at28/56, and finally an ISOLATED tail40→60 comparison at identical28/56/angular8 controls. The maximum relative change across every positive F/B number and all five energy moments, across all three scenarios, is3.011e-12 against the unchanged3e-6 gate. This tiny measured change is a numerical comparison at this smooth selected state, not a guaranteed 1e-12 physical or global integration accuracy.

The selected28/56/angular12/tail40 values are:

| Radiation | Forward number | Backward number | Net number | Net antineutrino energy |
|---|---:|---:|---:|---:|
| Empty | 1.117319281268152e-6 | 0 (empty incident radiation) | 1.117319281268152e-6 | 1.199461556398516e-5 |
| LTE | 1.045598158202876e-6 | 1.045598158202874e-6 | 1.58636e-21 (roundoff) | 1.97709e-20 (roundoff) |
| mu_anti=mu_eq-.4T | 1.065901791684378e-6 | 7.144953380713421e-7 | 3.514064536130361e-7 | 3.848695914568828e-6 |

Number units are fm^-3 s^-1, energy units MeV fm^-3 s^-1. For empty radiation the separately integrated positron, proton and neutron energy moments are1.056163598501366e-5, .001051744849666863 and .001053177829245836 MeV fm^-3 s^-1. Their signed matter change equals minus the antineutrino energy moment; direct event material-loss integration agrees. The largest scaled energy-ledger residual among qualified runs is1.31e-13. Nonequilibrium total entropy production is1.405625814452149e-7 kB fm^-3 s^-1; LTE entropy is numerical roundoff, not an inferred relaxation timescale.

The conditional Eanti=5MeV delta-cone calculation gives j=.3302219947096858 and chi=.9874077302846910 s^-1. Conditional20/12/16/8/tail40→28/16/24/12/tail60 refinement changes each by2.49e-12; comparison with the independently accepted response values differs7.36634e-8, within the unchanged3e-6 gate. The one-helicity DOS conversion and absolute inverse spin normalization thus pass a selected finite-temperature spectrum check. Vacuum source-polynomial reciprocity checks and malformed-input controls also pass.

The initial fourth refinement requested n36/e72/angular12/tail60 and exceeded its explicit110s budget. It produced NO completed fourth result; initial_refinement_partial.json and initial_refinement_timeout.txt preserve the three completed runs and the runtime exclusion. A cheaper separate isolated-tail run completed, and complete_refinement.py harvests only those three actual completed rows before adding new completed evidence. It does not manufacture or infer the stopped integral. Completed qualified experiments took11.14s,21.58s,74.05s and about22s, each below120s. Conditional runs took.90s and3.52s. There is also a preserved JSON-only NumPy-bool reporting failure after a completed guard-check integral; its corrected driver freshly reruns and saves explicit guard checks.

A late guard addition rejects an empty radiation-scenario list only. It does not modify valid occupation or event algebra; check_guard_identity.py replays a valid coarse case and compares all separate moments, net moments and density bitwise against its preserved pre-guard result. This is a control repair check, not a new accuracy claim. Positive backward coefficients are retained at finite FD occupation; ordinary/log representability is exposed, not floored.
