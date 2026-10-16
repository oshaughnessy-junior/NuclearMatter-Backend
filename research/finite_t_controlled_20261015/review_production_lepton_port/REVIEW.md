# Independent review: first production finite-T lepton utility candidate

**NOT READY TO INTEGRATE.** Selected warm/degenerate physics and package import pass, but four fresh resolved dilute forward states fail the unchanged3e-8 density gate. Preserve this initial disposition before an author repair. This review made no candidate, upstream, dependency, production, paper or git change.

Reviewed candidate manifest SHA256`025aa1a9426ef64f51a34f560d5e2de4f988ea37f0d5ce1a808c174b0d7ecfbc`, module`5dfe2dc902a2a98feab69981213855ac919b1268f0d16a5a9a4bb693fa2ab0d0`, base`ffb5cc72949f2e5401251847aef4c9c7a98dde78`. The exact five staged files and original report/checkpoint/manifest are copied into `reviewed_initial_candidate/`. No source fixes were performed. Independent evidence totals207 passing checks and4 physical forward failures; author60+13 remains separate historical evidence.

## Blocking forward finding

For a spin-two massive gas in the dilute limit, each particle population obeys

`n_MB = m² T K2(m/T) exp(mu/T) / [pi² HBARC³]`.

At mu=0 the exact FD relative correction is bounded by exp(-m/T), already below4.25e-18 for the first case. The selected production default electron mass is used, not the slightly different research mass.

| m/T | T (MeV) | Independently expected particle density (fm^-3) | Returned relative error | tail60->90 relative change |
|---|---:|---:|---:|---:|
|40|.01277497375|3.878797140295721e-29|1.6408275658785776e-4|-8.316224526772054e-5|
|100|.0051099895|8.358244596100698e-56|8.856983616949421e-4|7.51287900326858e-4|
|300|.001703329833333333|2.198589629850661e-143|8.968716881079253e-4|7.607965481915002e-4|
|600|.0008516649166666665|3.989337950304875e-274|8.996986084246483e-4|7.632022414258977e-4|

All are API-admitted, with positive resolved susceptibility and normal representable particle densities. They are not underflow states or exactly absent species. The report cannot relabel these failures as susceptibility underflow, model uncertainty or a successful warm-state gate. Tail refinement also fails to establish the requested selected accuracy.

`scaled_dilute_anchor.py` independently integrates original momentum in x=p/sqrt(mT), factors out exp(-m/T), and avoids subtracting E-m. Its64/96-node Gauss results agree with each other within3.89e-15 and with the scaled Bessel expression within4.67e-15, confirming the reference without relying only on a special-function identity. The production `_integrate` uses a fixed native epsabs=1e-18 with vector norm max. That permits absolute convergence on exponentially small integrands while missing relative population accuracy; this is the likely mechanism, inferred from the code and observed refinement. Tightening rtol alone does not remove the absolute scale. A numerically qualified evaluator or an explicit admission rejection is required before this public utility is integrated; the reviewer did not choose or implement the repair.

## Accepted selected physical and API scope

`probe.py` gives132 PASS. It uses actual `nmbackend` package initializers from the owned mirror, with24 imported package-module origins recorded and the exact candidate module overlaid. No stub package, fake JAX, campaign-driver import or altered production constants is used. A separate Git-blob reconstruction verifies all71 baseline package files; these provenance checks are counted separately from numerical physics. Only the five manifested staged paths are proposed for the eventual draft PR. The README change is a narrow link/example insertion; no package exports or existing cold sources are changed.

The physical measure is spin-two p²dp/[pi² HBARC³]. Net NUMBER is n_minus-n_plus; electric charge is its negative times e. Each flavor independently specifies total chemical potential or signed net number; antiparticle chemical potential is opposite within a flavor. No mu_e=mu_mu, baryonic neutrality or beta equilibrium is silently imposed. Energy includes both populations' rest energy, pressure and free energy use consistent MeV fm^-3, and entropy/Cv use kB=1. Documentation's cgs entropy/Cv conversion by1e39*kB is correct. These are independent ideal components, not a complete matter EOS or source-rest-Q prescription.

The response moments correctly give chi=A0/T, n_T=A1/T², mu_T|net=-A1/(T A0), Cv|net=(A2-A1²/A0)/T². The charge-sign transformations for negative chemical potential are consistent. Cv holds signed net number fixed while pairs equilibrate, not particle-only number. The stable thermal increment is rest-inclusive e(T,n_net)-e(0,n_net) without adding mass-Q twice.

New held-out states (T,mu,m)=(.73,-.29,M_E),(3.17,4.19,M_E),(13.7,91.3,M_MU),(.41,-117.2,M_MU) agree with an independent original-momentum rapidity-Gauss integration. It uses p=m sinh(y),48/80-node panels and independent energy-weighted covariance moments. The maximum selected forward/response discrepancy is9.91e-12; independent covariance refinement is1.88e-11. Production tail60->90/rtol2e-12->4e-13 checks pass at these selected states. A separate relativistic MB Bessel check at T1.7,mu=m-22T passes. Actual massive electron/muon tiny signed roots ±3e-29 at different temperatures reconstruct and agree with linear susceptibility. Fresh fixed-net Cv and mu_T finite differences follow matching unperturbed forward states.

A new actual cold package JIT call at n=.013 fm^-3 matches the combined finite-T electron/muon number and energy after explicit HBARC conversions; an actual cold energy Hessian is finite and positive. Existing cold JAX/Hessian behavior remains a separate source contract, not inherited by this CPU port.

The public finite-real/non-Boolean checks, T/m/rtol/tail restrictions and explicit unresolved-susceptibility exception are appropriate for their declared scope. Selected forged net/mu/nonfinite/Boolean state cases reject; detached asdict changes do not change the frozen state. A nonblocking API advisory: `thermal_energy_density` accepts finite changes to unused state.energy_density or state.cv fields, because it recomputes from T/mu/m and verifies only net consistency. Its mathematical output remains unchanged. The dataclass and doc's “returned state” condition are not provenance/authenticity protection; callers should not infer whole-state tampering detection. This is not an untrusted-code sandbox or a module-monkeypatch defense.

## Compatibility and remaining release gates

Executed environment: existing junior_tools Python3.12.12, NumPy2.3.5, SciPy1.15.2, JAX0.10.1. Every experiment was under120s. Python3.9 grammar parsing passes for all three staged Python files; built-in dataclasses and used NumPy primitives show no new obvious minimum-version syntax/API conflict. A primary [SciPy v1.7.0 source readback](https://github.com/scipy/scipy/blob/v1.7.0/scipy/integrate/_quad_vec.py) verifies the exact quad_vec keyword arguments and success/status attributes used here; raw readback is saved. This is static compatibility, not execution on Python3.9/NumPy1.21/SciPy1.7. Initial web direct-fetch cache misses are not substituted for verification; the owning repository source was subsequently read successfully.

Minimum-version runtime, admitted repository CI/full suite, wheel/build/install behavior, conda/pixi NumPy<2.3 environment and broad temperature/domain qualifications remain OPEN. No dependency change or remote PR was made. Warm-state numerical agreement does not certify interacting-model uncertainty, AD/JIT/vmap, cooling, full lepton matter closure or a cold-JAX replacement.

## Exact commands and preserved failures

From Genesis cwd, all using existing environments and owned outputs:

```
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/review_production_lepton_port/probe.py
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/review_production_lepton_port/dilute_probe.py
/Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/review_production_lepton_port/scaled_dilute_anchor.py
```

The first review-script version had a syntax typo and the first dilute reporter had a NumPy-Boolean JSON encoding failure. Their exact source versions are preserved as initial_*txt; these were reviewer implementation/reporting errors, not author physics failures or relaxed gates. The final independent four dilute failures remain fully recorded in `dilute_results.json`. `results.json` verifies all author bytes unchanged before/after its physical probes; final checkpoint independently verifies the complete author tree again. A repaired candidate requires a separate review; this frozen initial report must not be rewritten as acceptance.
