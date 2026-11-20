# Independent physics recheck: paired neutral-current sources

15 October 2026. This addendum reviews the repaired frozen `neutral_current/free_nc.py`, SHA-256 **b802b1799188dfb929ff44f2094db3e578009865400a5a548595f9f2c48d126c**. The original review and failed raw probe remain unchanged. Reviewer changed only new `review_physics/nc_repair_*` files. Account/window matched campaign guards at review start (5% used).

**Accept the paired-source numerical repair. The specific entropy-conditioning gate identified in `nc_review.md` is now closed at the original 2e-12 tolerance.** No physical kernel or rate normalization was changed. Existing restrictions on coarse energy grids, continuum evolution, free nonrelativistic matter and uncorrelated/static currents remain in force.

Executed `nc_repair_probes.py`; **11/11 independent checks pass**, recorded in `nc_repair_probe_results.json`. Source SHA was checked before and after execution. The preserved pre-repair source was loaded under a distinct module name; its SHA is recorded separately.

- Reproduce the exact held-out case: T=1.3 MeV, n=.0013 fm^-3, offset=17 MeV; irregular nodes `[3.5,5,7.5,11,18]`, weights `[.3,1.3,.8,2.1,4]`; occupations `[.91,.07,.63,.21,.005]`; CV=.2, CA=.7; angular order 40.
- The preserved implementation reproduces the original **2.16747e-10** entropy-identity error, failing the unchanged 2e-12 target. The repaired source gives **2.22045e-16** against the independently assembled unordered-pair nonnegative entropy formula.
- Independent unordered-pair bath heating cancels the returned radiation-energy source to **4.41e-16 relative**. Number residual is **1.51e-23** on the gross-flux scale.
- Angular weight, log differential kernel, blocked differential kernel and bare static cross-section function ASTs are unchanged. The complete held-out transition-rate matrix is **bitwise identical** before/after repair. Thus this repairs cancellation of identically cancelling/self fluxes rather than adjusting the collision physics or its detailed balance.
- Two new held-out valid-control incident-energy moment calculations, neutron and proton at 7.3 MeV on the same bath, are **bitwise identical** across the old and new sources. Added input validation does not alter these physical calculations.

The repair accumulates each unordered off-diagonal `Jji−Jij` into opposite bin signs and omits the self-transition contribution that is identically zero. This is the correct stable assembly for the same discrete operator. It does not redistribute unresolved continuum spectral weight into represented nodes, invent leakage, or introduce a new physical scattering correction.

The author now reports 189 checks; the separate numerical review reports its own checks. This recheck neither substitutes those counts for the original counterexample nor discards the failed evidence. `nc_initial_probe_failure.json`, the old `nc_probe_results.json` and `nc_review.md` retain the original result. The present pinned addendum records exactly which numerical gate has been repaired.

Publication may describe the selected free-NC kernel and conservative discrete example with this repair and its limits. It must still avoid converged continuum heating/spectral-evolution claims, interacting density/spin-response claims or any suggestion that the external finite-temperature GM1 forward gate has closed.
