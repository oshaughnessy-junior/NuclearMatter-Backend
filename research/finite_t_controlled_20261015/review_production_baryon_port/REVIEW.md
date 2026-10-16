# Different independent review: particle-only baryon reference

**ACCEPTED as a selected scalar CPU ideal-gas reference:325 checks PASS, no candidate failures.** Counts are126 pre-candidate independent-oracle checks,193 candidate forward/response/domain/provenance checks and6 additional Maxwell-response checks. No author source or frozen evidence was changed. This does not authorize automatic production publication; packaging/scope, supported-version runtime and admitted repository CI remain separate gates.

Candidate source f4c007f400df346811d4ce759ee8d96d6b74170c7ef1d0c4f5bc063f8018f65b; manifest199716b691d88a4fc9b83ed7f8e8b48e36fbff16fcc9b1e8684d18d0922fc523. VALIDATION_LOCK.json and pure original-momentum/Bessel oracle were prepared before candidate inspection or import. Candidate comparisons began only after the author sent exact frozen hashes. The full candidate tree is byte-identical before/after the main probe; every manifested path verifies.

## Model, masses and units

The component is one spin-two relativistic ideal **particle** FD gas. Its positive particle number is not a particle-minus-antiparticle density; zero/negative density rejects. Total chemical potential can legitimately be negative while particle population stays positive. No antiparticle population, pair-equilibrium completion, omission veto, neutrality, baryonic composition relation, mean field or nuclear interaction is supplied. The documented mathematical particle-only model remains defined where a physical full pair gas would require additional species; physical applications must justify that omission separately.

Mass is fixed in all response derivatives. Parameterizing this ideal component by a supplied quasiparticle mass does not add density/temperature rearrangement terms or select a weak-current/spinor mass. Direct kinetic energy integrates E−m and total energy is m*n+kinetic, avoiding a cold-rest subtraction. T,total mu,m are MeV; n,s,Cv are fm^-3 with kB=1; energy,pressure,free energy are MeV fm^-3; chi and n_T are fm^-3MeV^-1; mu_T is dimensionless. Documentation and example consistently distinguish these from cold geometric/fm^-4 units, a full EOS, cooling, transport or JAX/AD.

Exact NMB defaults are M_N939.565421 and M_P938.272088MeV. The earlier capture chain uses939.56542052/938.27208816MeV. Main preregistered density cases use explicit NMB masses, and original capture-source parity uses the explicit campaign masses separately; no silent rest-mass convention change is accepted. Both original capture chemical potentials and direct kinetic energies agree under those exact matched masses.

## Independent forward values before derivatives

The oracle integrates original momentum p with independent scalar QUADPACK per moment, panels at actual momentum Fermi/thermal scales, finite90T tail and100T refinement. It retains actual FD occupations and stable entropy softplus factors. It does not import the candidate or substitute a classical gas. A separately derived16-term alternating fugacity/Bessel series checks dilute number,pressure and rest-inclusive energy. Tail/series/Euler/positive-variance oracle checks pass before the candidate is used.

Nine preregistered points include physical neutron/proton T2MeV capture densities7e-8/3e-8fm^-3 under NMB masses, a new neutron T1.37,n4.6e-5, dense/cold neutron T.041,n.16 and proton T.073,n.27, neutron T3.4,n.12 and three new dilute states with eta−31.7,−15.3,−4.2. For each, the unperturbed forward n,kinetic,E,P,s,F gate passes before response work; maximum bulk discrepancy2.85e-13. Density roots reconstruct positive targets to1e-9. All response fields agree with independently integrated original-momentum moments within3.49e-11 (gate2e-8).

The response oracle uses w=f(1−f), A0,A1,A2 with actual E−mu and computes the covariance. The candidate instead uses a second positive centered kinetic-energy variance integral, with no large rest/dilution moment subtraction. Agreement plus positivity checks support that fixed-particle-number Cv implementation at these states, not arbitrary-domain positivity or differentiability.

The first independent oracle entropy implementation rounded f to1 before evaluating vacancy entropy and emitted roundoff/subdivision warnings despite passing126 comparison gates. Its exact source/results/warning are preserved. Before candidate use, stable exact softplus/log-occupation entropy removes those warnings; the final126-check oracle has empty stderr. This was an independent numerical-fixture repair, not a candidate fix or relaxed gate. Only the final resolved oracle checks are counted once.

## Thermodynamic and limiting checks

Euler/free-energy/rest-plus-kinetic identities pass with clearly rest-inclusive energies. At three new warm/dense states, separate central finite-difference ladders with two steps and Richardson extrapolation test re-solved fixed-number dkinetic/dT=Cv and dmu/dT, plus fixed-mu dP/dT=s,dn/dT, fixed-T dP/dmu=n and dn/dmu=chi. Maximum relative ladder discrepancy1.41e-10 against2e-6. These are thermodynamic response checks after forward agreement, not gradients through adaptive quadrature or the root.

Six additional checks directly differentiate entropy and kinetic energy with respect to mu: ds/dmu|T=dn/dT|mu and dkinetic/dmu=(mu−m)*chi+T*n_T. Raw step ladders and extrapolations are retained. This probes the independent entropy/response construction rather than relying only on a rest-dominated Euler cancellation.

For both dense cold states the leading relativistic Sommerfeld Cv=n*pi²*E_F*T/p_F² differs by−1.48e-6 and−2.35e-6, within the explicitly approximate2e-5 limit gates. The dilute NR Cv/(1.5n) differs by6.12e-4, compatible with finite T/m corrections and its declared.002 limit gate. Neither asymptotic form replaces actual FD integration. These limits are not exact finite-temperature formulas or broad cold-domain certificates.

## Admission, real package and omission boundaries

The test uses the real frozen NMB package/mirror initializer, with no namespace stubs. Fresh negativeTOTALmu states and very dilute finite populations pass independent momentum comparisons. Positive n=1e-300 and1e-310 reconstruct and have positive Cv; only root/positivity is claimed for these tiny points, not relative subnormal output fidelity. Invalid/nonfinite/Boolean/array scalar inputs, zero/negative particle number, invalid mass/T/tolerance/tail reject. Finite controls causing unrepresented population, interval or density-root range raise ArithmeticError instead of returning a density floor or exact absence. No signed-net-particle interpretation is introduced.

The public module supplies ParticleState,state_mu,state_density only, with no pair or thermal-subtraction API. Centered Cv, fixed-mass partials, independent species and direct kinetic energy are documented consistently. All current cold APIs and baseline package files remain separate; the author reports36 candidate+13 cold tests and71 base blobs, but those counts are not relabelled independent execution here.

## Preservation and remaining gates

The five manifested paths, source/report/checkpoint and all main before/after author bytes verify. Original source snapshots are copied in reviewed_candidate/. Raw forward states, independent oracles, both derivative ladders, guards, initial oracle warnings and source hashes are retained. No source correction, dependency/install, build, production, paper, git/branch/PR or manual CI action occurred.

Existing junior_tools runtime is used; its NumPy2.3.5 remains outside the separately pinned validation environment's<2.3 range. Minimum/supported runtime, wheel/install packaging, full admitted repository CI, uniform parameter-domain/error certification, interacting EOS closure, density-dependent effective-mass thermodynamics, antiparticle completion, weak-current selection and AD remain open. Selected free-component acceptance is not a physical dense nuclear-matter or merger/cooling uncertainty claim.

Commands from Genesis:

```
/Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/review_production_baryon_port/check_oracle.py
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/review_production_baryon_port/probe.py
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/review_production_baryon_port/check_maxwell.py
```
