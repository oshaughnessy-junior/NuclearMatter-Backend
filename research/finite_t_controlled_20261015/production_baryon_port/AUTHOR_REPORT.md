# Particle-only baryon reference: staged author candidate

**Frozen for a different independent review; not accepted for integration.**
Exactly five owning NMB paths are staged:src/nmbackend/reference/
finite_t_baryons.py,tests/test_finite_t_baryons.py,examples/finite_t_baryons.py,
docs/finite_temperature_baryons.md and a narrow README section. Fresh main
remainsffb5cc72949f2e5401251847aef4c9c7a98dde78; existing canary changes are
preserved. No actual checkout, lepton candidate/mirror, production source,
branch, PR, commit, dependency or external environment was modified.

## Model and interface

The separately imported CPU utility supplies ParticleState,state_mu and
state_density. Its strictly positive particle_density is NUMBER, not net
particle-minus-antiparticle density. Temperature,total mu,mass are MeV;
n,s,Cv are fm^-3 with kB=1; E,K,P,F are MeV fm^-3. Both total mu and energy
include rest mass. K is a direct integral of E−m, not a subtraction of large
rest energies. There is no cold thermal-subtraction API, composition/neutrality
constraint, assembled EOS or energy inversion added for its own sake.

All responses explicitly hold the supplied positive mass fixed. Arbitrary
constant declared bare/quasiparticle masses parameterize an ideal gas only;
no density/T mass derivatives, rearrangement, mean fields, calibrated
interactions or weak-current mass convention is supplied. Antiparticles are
omitted from this mathematical particle-only model. There is no diagnostic
veto that silently excludes high-T states, and no claim those omitted species
are negligible everywhere. It is not a complete pair gas or physical high-T
matter EOS. Existing cold/JAX/EOSTable/cooling APIs and exports are unchanged.

Defaults import exact NMB M_N=939.565421,M_P=938.272088 and HBARC. Campaign
capture masses939.56542052/938.27208816 differ slightly. Explicit same-mass
arguments are used for comparisons with the accepted physical source EOS;
neither default nor physical convention is silently replaced.

## Numerics and primary forward gate

The ideal FD measure/equations match the accepted particle reference, with
new scaled dimensionless kinetic integration rather than native amplitude
floors. Occupation and dimensionful scales restore together in logarithms;
all adaptive absolute tolerances are zero. Entropy is direct and positive.
Cv uses a second centered positive variance integral, avoiding subtraction of
large second moments. The logarithmic density root retains tiny positive
populations, and final represented total mu must reconstruct n to1e−9.
No floor, fit, classical fallback or surrogate is introduced. Finite-real/
non-Boolean guards and unresolved integral/positive-response guards remain
strict. The proof/units are in evidence/NUMERICAL_CONTRACT.md.

**48 primary forward checks PASS** at the actual accepted transparent capture
baths T1,nB1e−7;T3,nB1e−7;T2,nB1e−6, for both physical n/p masses. Fresh
accepted physical EOS values and new states compare particle number,K,E,P,s,F
and density roots/total mu BEFORE any response comparison is launched. This
is selected numerical parity with an accepted reference, not a new physical
model or new capture-source integration.

## Further executed checks

**36 candidate tests PASS**, plus **13 unchanged cold derivative/JIT/Hessian/
unit tests PASS**,49 total in4.88s. The initial32-test run also passed and
its raw output is retained. Separate original-momentum quadratures gate
forward outputs before susceptibility,Cv,n_T and mu_T at six selected dilute,
warm,degenerate,relativistic and declared-mass states. Other tests cover:

* Relativistic scaled-Bessel dilute moments at m/T40,600 with eta−35,−600,
  including resolved very small populations and all thermodynamic/responses.
* Positive roots down to1e−280fm^-3, cold relativistic/Sommerfeld and NR
  limits, Legendre and pressure derivative identities, Maxwell relations,
  and decreasing fixed-density temperature steps after forward gates.
* Independent momentum/scaling checks at T1e−24,1e−9,1e6MeV with explicit
  proportional masses, zero absolute assertion floors and tail/tolerance
  refinement. These are parameterized calls, not mass-varying responses.
* Strict scalar/sign/Boolean admission, immutable states, ordinary unresolved
  population rejection, and a total-mu offset that cannot be represented and
  therefore fails the required forward reconstruction.

No numerical gate failed or was relaxed. The initial orientation attempted
a nonexistent review_relativistic_free_eos path; the actual accepted
review_relativistic_eos/REVIEW.md was read before implementation. This read
path diagnostic did not execute scientific code or modify any evidence.
All original accepted-reference failures remain untouched, not relabeled.

The example emits finite named-unit JSON,stderr empty. The owned mirror uses
all71 exact pinned baseline package Git blobs and real package initializers;
24 actual package modules resolve inside that mirror, with the candidate
separate. The existing lepton test mirror remains untouched. The three staged
Python files parse with Python3.9 grammar. Existing execution environment is
Python3.12.12,NumPy2.3.5,SciPy1.15.2,JAX0.10.1,Diffrax0.7.2,pytest9.0.3.
No installs or fake packages. Syntax is not minimum-version execution; NumPy
is outside the separate conda/pixi<2.3 validation pin. Only APIs available
under NumPy>=1.21/SciPy>=1.7 are used (no scipy.special.log_expit).

## Remaining gates and scope

A DIFFERENT physics/numerical/API review is required before any root-owned
production draft publication. Normal admitted repository CI/full suite,
minimum/supported versions, pin-environment and packaging remain open. Only
the five manifested staged paths belong in any future scoped draft; research
snapshots, test mirrors and reports are evidence, not production dependencies.

No uniform parameter-domain/relative-subnormal precision certificate,
physical uncertainty, full pair matter, composition/rate/transport/cooling
inventory, interacting EOS, AD/JIT/vmap or differentiation through adaptive
quadrature/root is claimed. Ordinary subnormal/zero K or P is a representation
limit, not exact absence; unresolved positive n/chi/Cv rejects. Tail/tolerance
checks are selected numerical witnesses, not an analytic global tail bound.
