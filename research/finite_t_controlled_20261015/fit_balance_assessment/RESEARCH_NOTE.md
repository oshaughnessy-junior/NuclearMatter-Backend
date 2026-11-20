# Selected cancellation/root assessment of optional capture-moment fits

**Author-frozen assessment; a different independent review is required.** This is an explicit new experiment on cancelled source/root accuracy. It does not change the optional fit API, its individual-moment acceptance or the detailed physical reference/default. There are112 passing numerical checks and4 passing preregistered model-assessment gates. No gate failed, no coefficient/basis/domain/gate was retuned and no production or paper file changed.

## Preregistered experiment and source identity

LOCK.json was written2026-10-16T02:02:59.098619Z before new source calls. It selects exactly(T MeV,nB fm^-3)=(1,1e-7),(2,1e-6), brackets Ye[.25,.45] and[.22,.35], and fixes numerical source refinement3e-6, target FD density3e-8, EOS root1e-10 and approximate root net/gross1e-9. Model criteria are |deltaYe|<=.001 and |actual detailed net|/gross<=.001, chosen before data. They are practical selected-state assessment criteria, not a uniform error guarantee.

The candidate source is actual beta decay PLUS optional fitted EC/PC separate positive moments. Every approximate root trial solves fresh physical free relativistic neutron/proton and neutral pair-electron EOS, calls the fixed optional API, and executes actual full-current beta event moments. No beta fit or saved beta rate is substituted. EC/PC fit source is approximation.py9020fb5d..., admitted by the separate repaired-consumer review412ede4320ad2c6b9befb74705b17a4e66f1089bc025080a2f83e0f31fee2eb3. Coefficients/source/qualification remain unchanged.

Both candidate roots lie inside the actual EOS intersection |mu_e/T|<=3 and baryon eta<=-5. Original T2,nB1e-7 root Ye~.55604 and T3,nB1e-7 root Ye~.54901 are outside the fit's Ye<=.5 boundary; explicit calls reject them. They are not extrapolated, assessed or silently clipped.

The physical current uses the same free masses, total chemical potentials, GF/Vud/gA, full weak magnetism/pseudoscalar/q² form factors and bare weak mass as the accepted detailed sources. Both particle-hole baryons and e± Pauli factors are retained in the detailed event references. No alternative EOS/current family or medium/correlated correction is inferred.

## Results at the two candidate roots

| T MeV,nB fm^-3 | Accepted detailed reference Ye | FitEC/PC+actualbeta root Ye | deltaYe | Actual detailed net/gross at candidate | Actual dYe/dt s^-1 |
|---|---:|---:|---:|---:|---:|
|1,1e-7|.3850567101958036|.3850564232269156|-2.8696888804e-7|+1.1519207892e-6|+4.4490798958e-7|
|2,1e-6|.2786277578701185|.2786271184968538|-6.3937326472e-7|+3.1266893961e-6|+3.7148714306e-5|

All four locked model gates pass. **The actual detailed net is nonzero at the approximate root**, retained with its sign and absolute dYe/dt. These are not exact detailed stationary compositions. Comparing a relative error against zero source would be meaningless; gross-normalized residual and absolute deltaYe are the two separately declared measures.

At each candidate, both actual full beta+PC+EC profiles are executed. Every separate number, neutrino-energy, charged-lepton-energy, proton/neutron-energy and direct material-loss moment refines, maximum1.095e-8 against3e-6. All physical EOS and incoming-particle-density checks, rest-inclusive species energy ledgers and electron-lepton ledgers pass. These two quadratures share the physical kernels; numerical agreement is not nuclear-model independence or uncertainty.

Individual fit errors at these candidate compositions are retained as diagnostics, separately from root gates:

| T,nB | R_EC fraction | Q_EC fraction | R_PC fraction | Q_PC fraction |
|---|---:|---:|---:|---:|
|1,1e-7|-7.64797e-7|-7.70184e-8|-3.07367e-6|-2.55955e-6|
|2,1e-6|+1.90899e-6|+9.76909e-7|-4.34461e-6|-3.13597e-6|

No new individual1% gate is added after observing these results. The separate fit review's four-point/author-heldout qualifications remain their own evidence. The accepted stationary-allowed assessment's20 model failures also remain failures; this comparison does not erase them.

## Composition, energy and thermostat bookkeeping

For both exact and approximate sources use

`R = Rbeta + RPC - REC`, `dYe/dt = R/nB`,

and positive escaped energy

`Qescape = Qbeta + QPC + QEC`.

Beta and PC create one material net electron each; EC consumes one. Escaped radiation electron-lepton source is REC-Rbeta-RPC and cancels material R. One active neutrino helicity per event is already included; no extra DOS or spin factor multiplies integrated moments. The approximate moment API supplies no separate collision species-energy moments; only the detailed full source has those independently accumulated ledgers.

Temperature and baryon density are prescribed. Rapid EM pair equilibration maintains the FD e± bath, with an external thermostat; photons and photon transport are not evolved. At exact composition balance the thermostat supplies Qescape, but at these approximate roots the actual source can still drift. To avoid silently dropping the rest/composition energy term, the fixed-T required external power is

`H = Qescape + u_Y * R`,

where u is rest-inclusive energy per baryon and

`u_Y = (mu_p - T mu_p,T) - (mu_n - T mu_n,T) + (mu_e - T mu_e,T)`.

This reuses the accepted free-energy thermodynamic responses at fixed species/net density, not a derivative through the fit. It is independently checked by a finite composition difference of the actual EOS energy, with absolute discrepancies3.82e-7 and1.91e-7 MeV against2e-5. No parameter AD/gradient claim follows.

At T1, u_Y=1.0097953600MeV, actual Qescape=2.0045759817e-7 and actual H=2.0045764309e-7MeV fm^-3s^-1. At dense T2, u_Y=2.8559960262MeV, actual Qescape=1.2058970540e-4 and actual H=1.2058981149e-4. Fit-source escaped energies are2.0045730400e-7 and1.2058956272e-4 respectively. All quantities include physical rest energies through the EOS/events. Adding a separate neutron-proton reaction Q or2me pair replenishment term would double count. This is a prescribed thermal bath assessment, not a cooling evolution or energy equilibrium.

## Evidence, reproducibility and limits

The entire experiment takes47.64s, with4 combined detailed-source calls total. Each bath has8 fresh fit+physical-beta trial evaluations. Detailed moments are evaluated only at the candidate roots; immutable previously reviewed detailed roots supply the reference compositions. No multidimensional detailed root search or broad grid is rerun.

Exact command from Genesis, existing environment and no installs:

```
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/fit_balance_assessment/assess.py
```

Completed results refuse overwrite. For independent reproduction use a new owned sibling output directory containing only the source and exact LOCK.json; all frozen input paths remain siblings in the campaign. Never rerun into the completed evidence directory. Full source/moment trials, roots, brackets, separate controls, model gates, outside-domain rejections and input hashes are in results.json; partial checkpoints are retained. The current source, lock, evidence and notes are hashed in MANIFEST.json. All pinned external source/fit/review/reference files verify unchanged after execution.

Acceptance is pending a different independent review. Even passing these two cancellation/root tests does not permit a general net-rate/root API, uniform sign guarantee, whole-domain certificate, arbitrary radiation closure, interacting matter, missing weak-channel inventory, long-time trajectory, derivative/AD or physical uncertainty claim. Detailed physical quadrature remains reference/default and fitted individual moments remain optional.
