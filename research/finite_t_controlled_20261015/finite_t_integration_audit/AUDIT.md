# First finite-temperature NMB tooling PR: independent feasibility audit

**Disposition: a small, installable CPU lepton-state utility is ready to implement as a draft production PR. It is not yet a tested/merge-ready port. A replacement for the JAX cold EOS or a general finite-temperature matter API is not independently qualified.** There is no physics-source merge conflict at the inspected head. The first PR can deliver callable tooling, tests and a short example without importing campaign archives, changing existing EOSs or waiting for weak-rate/cooling integration.

This audit is read-only with respect to NMB and all existing research leaves. It makes no new physical acceptance claim and runs no numerical experiment. Readback/provenance evidence is stored in the two adjacent GitHub JSON files.

## Pinned repository state and instructions

The research physical-source baseline remains **25e68182f726e5bd9c0b372c9a667536940fbd07**. Fresh GitHub `main` at audit start is **ffb5cc72949f2e5401251847aef4c9c7a98dde78**, the PR50 identity-only scientific-review canary merge. GitHub compare reports main four commits ahead and lists exactly one changed path: **`.github/workflows/private-review-dispatch.yml`, added, 202 lines**. No physics source changed. Fresh `src/nmbackend/eos/leptons.py` has the same Git blob **7cd4565470dafb48967546897fc99cc3d8935ad8** at both commits. A future branch must start from current main and preserve the workflow; research baselines must not be silently repinned.

GitHub reads of `AGENTS.md` at root, `src/`, `src/nmbackend/`, `src/nmbackend/eos/`, `src/nmbackend/reference/`, `tests/`, `docs/` and `examples/` all return 404. This establishes absence at those exact paths, not an instruction blocker. No `/AGENTS.md`, `/Users/AGENTS.md`, `/Users/rossma/AGENTS.md` or `/Users/rossma/NuclearMatter-Backend/AGENTS.md` was available in the checked local ancestor locations. Genesis instructions govern these audit artifacts. NMB README, architecture, source, test and dependency files were fetched directly from the owning GitHub repository rather than inferred from an unrelated local checkout.

## Existing production contract

[`eos/leptons.py`](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/ffb5cc72949f2e5401251847aef4c9c7a98dde78/src/nmbackend/eos/leptons.py) is a **cold JAX component**. Its actual entry point is `lepton_sector(n_charge, muons=True, n_bis=50)`, returning `(mu_e, n_e, n_mu, eps_lep)`. Inputs/outputs use mu and masses in **fm−1**, number in fm−3, and energy density in **fm−4**. Chemical potential and energy include rest mass. It imposes mu_e=mu_mu and n_e+n_mu=n_charge, floors requested charge at 1e−30, and has a cold muon threshold. It is not an arbitrary-composition pair plasma API.

RMFT, metamodel, DDRMF and Skyrme import this shared cold function. [`tests/test_lepton_derivatives.py`](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/ffb5cc72949f2e5401251847aef4c9c7a98dde78/tests/test_lepton_derivatives.py) checks density reconstruction, the energy gradient, the **energy Hessian**, vector independence and JIT. Replacing it with a SciPy quadrature or first-order callback would remove an existing contract. The cold `EOSTable` is explicitly barotropic, with no temperature/composition coordinates; `Composition` is beta-equilibrium `(n,x_p,x_e,x_mu)` on that cold grid. These are unsuitable containers for a new independent thermal state without a separate interface decision.

Existing cooling heat is another distinct contract: [`cooling/physics/heat.py`](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/ffb5cc72949f2e5401251847aef4c9c7a98dde78/src/nmbackend/cooling/physics/heat.py) follows NSCool's degenerate heat-capacity prescription, with local **kelvin**, kF in fm−1, mass in MeV, and output erg cm−3 K−1. It intentionally retains source-rounded constants and pairing prescriptions. The new finite-T pair-gas Cv must not silently replace that source-labelled parity calculation.

## Transferable accepted reference

The candidate is `thermodynamics/lepton_reference.py`, SHA256 **f87ea4bcf8830174995bf9c7061a943052c2d3be8d4386befe3c90215ef97ec9**, together with its selected analytic/finite-difference tests and independent numerical review. The current author evidence has 86 checks. The original independent review's near-pair-symmetry density-root failures are retained; its repair addendum independently accepts the scale-aware dimensionless inversion/residual repair, with 30 mixed numerical checks. No filled state rectangle or extreme low-temperature domain was accepted.

This is an originally written ideal relativistic spin-two particle/antiparticle gas, not copied NSCool/Fortran code or a model calibrated to weak-source totals. The reusable material is the FD occupations, stable signed net-number subtraction, kinetic-energy integration panels, free-energy/entropy definitions, positive susceptibility moments, fixed-net-density response identities, scaled density inversion and stable thermal-energy increment. It requires no paper archive, source-pin registry, accepted-review file, radiation closure, nucleon model, external table or GPU. Research orchestration/JSON reporting and all weak/cooling factory machinery should stay outside the port.

## Concrete first PR

Suggested branch **`feature/finite-t-lepton-reference`**, based on a freshly checked current main. Suggested title: **“Add finite-temperature ideal lepton pair states as a CPU reference utility”**. No branch or PR is created by this audit.

Proposed exact file list:

| Path | Scoped change |
|---|---|
| `src/nmbackend/reference/finite_t_leptons.py` | Installable NumPy/SciPy scalar state evaluator and inversion, adapted from the accepted reference; immutable `LeptonState`; explicit module import |
| `tests/test_finite_t_leptons.py` | Analytic limits, independent momentum oracle, inversion/response/input regressions and isolated cold compatibility checks |
| `examples/finite_t_leptons.py` | Small executable example calling electron and muon states independently and printing named units; no production output fixtures |
| `docs/finite_temperature_leptons.md` | Input/field/unit table, exact constraints, selected benchmark ladder, CPU/no-AD/underflow limitations |
| `README.md` | Short link and runnable example command; distinguish the new utility from the cold JAX EOS/TOV and NSCool paths |

The existing `nmbackend.reference` namespace already contains explicit, out-of-hot-path SciPy reference solvers. It is the smallest architecture-compatible home. Do not eagerly re-export from `nmbackend/__init__.py`, alter `eos/__init__.py`, replace cold `leptons.py`, extend `EOSTable`, or change cooling kernels. Explicit use is `from nmbackend.reference.finite_t_leptons import state_mu, state_density, thermal_energy_density`. This is reusable installed tooling, although it deliberately does not yet enter the inference/AD hot path.

Proposed signatures, for review rather than implementation here:

```
state_mu(temperature_mev, chemical_potential_mev, mass_mev=M_E,
         *, rtol=2e-12, tail=60.0) -> LeptonState
state_density(temperature_mev, net_density_fm3, mass_mev=M_E,
              *, rtol=2e-12, tail=60.0) -> LeptonState
thermal_energy_density(state, *, rtol=2e-12, tail=60.0) -> float
```

Use keyword `mass_mev=M_MU` to request the muon component. Each flavor has its **own** mu or signed net-density input at the common chosen T. Calls do not impose mu_e=mu_mu, beta equilibrium or charge neutrality. Pair equilibrium within one flavor imposes mu_antiparticle=−mu_particle. At net zero, mu=0 and the pair plasma remains; “net muon density zero” does not mean “muon species disabled.” A later matter wrapper can impose n_e,net=nB Ye, n_mu,net=nB Ymu and n_p=n_e,net+n_mu,net, with explicit baryon-domain guards. That wrapper is outside this first PR.

Use NMB's existing HBARC/M_E/M_MU constants in the production default. The reference electron mass is .51099895069 MeV while NMB M_E is .510998950 MeV: regenerate port expectations at the explicit production mass rather than claiming bitwise default-mass parity. Cold internal conversion is mu_MeV=HBARC*mu_fm−1 and epsilon_MeV/fm³=HBARC*epsilon_fm−4. No hidden mass/Q shift belongs here.

The immutable state's contract should preserve explicit meaning: T, total mu and m in MeV; signed n_net=n_minus−n_plus and separate nonnegative particle/antiparticle densities in fm−3; **rest-inclusive total pair energy**, pressure and free energy in MeV fm−3; entropy and Cv at fixed **net** density in fm−3 with kB=1; chi=dn_net/dmu|T and dn_net/dT|mu in fm−3 MeV−1; dmu/dT|n dimensionless. The stable thermal increment is energy at fixed signed net density relative to that component's cold state, not total-matter heat. Analytic thermodynamic response fields are not a promise that `jax.grad`, Hessians, JIT or vmap work on this CPU dataclass.

## Meaningful forward gates before merging that PR

1. **Ported forward evaluator against two different references.** Preserve the accepted analytic limits and add an independent scalar original-momentum oracle directly in tests; compare e and muon particle/antiparticle densities, total energy and pressure at both signs of mu, mu=0, warm pairs and degenerate states. A copied fixture alone is not independent validation.
2. **Net-charge inversion.** Positive/negative/zero requests, the preserved tiny signed massless requests down to1e−30 fm−3, requested-versus-reconstructed residual at1e−9 or tighter, and monotone positive susceptibility. Do not restore a fixed absolute MeV root tolerance or the cold1e−30 density floor.
3. **Limits and units.** Exact massless formulas; nonrelativistic Maxwell–Boltzmann density/P/nT/Cv; massive cold formulas after the explicit HBARC conversion; Sommerfeld Cv and stable thermal increment. A cold finite-T error is O(T²), so test that limit with a stated tolerance rather than expecting exact T=0 equality. T=0 remains rejected by the new API; the old cold API remains the cold entry point.
4. **Response identities separately from AD.** Euler e+P=Ts+mu*n_net, free-energy identity, Maxwell ds/dmu=dn/dT, finite differences of n(mu,T), Cv|n and dmu/dT|n. Use stable thermal increments/Sommerfeld checks where subtracting large cold energies hides the thermal difference. No derivative work should precede a matching forward state.
5. **Admission and representation.** Explicit finite-real, non-Boolean T>0/m>=0/mu/signed-n validation and finite rtol/tail bounds; reject failed quadrature, unresolved root or susceptibility underflow. Retain tests for the documented cold/dilute massive-pair underflow exception. Do not floor a positive tiny population or mistake an ordinary underflowed antiparticle density for an exactly forbidden species. Relative accuracy of unrepresented populations is not qualified; the first PR must state that, rather than imply a new log-density API. These stricter production guards require review and targeted tests; the research source's `map(float)` is not itself a complete public input contract.
6. **Separate numerical refinement.** Selected finite-T states need independent tail60→80 and tolerance refinement, with recorded checks for every field used. Passing thermodynamic identities alone cannot establish quadrature accuracy.
7. **Existing contracts and supported environments.** Run new pytest tests plus unchanged `tests/test_lepton_derivatives.py` and relevant existing cold EOS/unit tests, then the repository's admitted CI path. Existing tests use parametrized pytest and `numpy.testing.assert_allclose`; the cold Hessian/JIT tests remain untouched. Do not manually bypass CI budget admission or modify the canary workflow.

[`pyproject.toml`](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/ffb5cc72949f2e5401251847aef4c9c7a98dde78/pyproject.toml) already declares NumPy>=1.21, SciPy>=1.7 and pytest>=7 in the test extra; no new dependency is needed. It advertises Python>=3.9. The validation environments use Python3.11 or>=3.11,<3.13, NumPy>=1.24,<2.3 and SciPy>=1.10; full admitted CI covers Python3.11/3.12/3.13. Research was checked in newer/different environments, not all supported package minima. Production PR authors must exercise a compatible declared environment and either preserve the existing minimums or explicitly justify/update all affected dependency declarations. An audit cannot mark that validation done. Full-validation dispatch is budget-controlled, so a numerical notebook run is not a substitute for its normal admission.

## Exact readiness boundary

**No unresolved physics or architecture issue prevents writing this scoped CPU-tool PR now.** Its remaining merge gates are the actual packaged implementation, strict input/underflow policy tests, explicit constants/unit adaptation, independent forward/cold-limit regression and supported-environment CI. Nothing in this audit is evidence those implementation gates have run.

**A native differentiable finite-T EOS PR is blocked at a different gate:** there is no production JAX finite-T evaluator/root implementation independently checked against these forward values, with qualified JVP/VJP and required higher derivatives; `EOSTable` and cold `Composition` also do not represent independent T/Ye/Ymu. A first-order callback or scalar adaptive SciPy code cannot inherit the cold lepton energy-Hessian contract by being inserted behind the old name. No arbitrary-composition interacting baryon free energy, physical weak-source consistency or full cooling inventory follows from the lepton utility. Those belong to subsequent, separately scoped work.
