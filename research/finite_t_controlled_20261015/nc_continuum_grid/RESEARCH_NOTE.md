# Selected continuum neutral-current redistribution and resolving energy grids

Research checkpoint,15October2026; independent review required. This leaf establishes **selected-incident-energy continuum-integral convergence** for the accepted free-NR neutron NC kernel, with final-neutrino blocking and gain/loss. It compares actual outgoing-energy quadratures with physical energy-node grids. It does not evolve radiation or matter, compute a global double-energy heating integral, or qualify an interacting opacity/full transport closure.

## Physical ansatz and equations

The frozen `neutral_current/free_nc.py` supplies the physically normalized leading free, unpolarized neutron density/spin response with matched FD nucleons, spin2, common939MeV kinetic mass, quadratic recoil and nuclear Pauli blocking. There are no interactions, RPA/spin correlations, weak magnetism or relativistic-current corrections. Offset is0 for this diagonal response. No frozen upstream source is changed. The kernel convention is the same differential rate as [Bedaque et al., Eq1](https://arxiv.org/html/1801.07077); the occupation collision measure follows [Burrows/Reddy/Thompson](https://arxiv.org/html/astro-ph/0404432). Those primary references were revisited; their interacting calculations are not implemented here.

WithE incoming, E' outgoing,omega=E−E', q²=omega²+2EE'(1−c), letK(E→E',c) have units s^-1MeV^-1, before final neutrino blocking. It **already includes outgoingE'² phase space**. Thus outgoing-energy quadrature weights are dE', without another neutrino DOS. The one-helicity number measureg(E)=E²/[2pi²(hbarc)³] enters the reverse gain ratio:

    C(E)=integral dE' dc [(E'²/E²)K(E'→E,c)f(E')(1−f(E))
                         −K(E→E',c)f(E)(1−f(E'))].

Independent direct reverse-kernel evaluations verify

    K(E'→E)/K(E→E')=(E/E')² exp(−omega/T).

This is scattering redistribution, not an emission/absorption scalarj/chi. Nuclear chemical potentials cancel from this detailed-balance exponent but remain in the response amplitude/blocking. Any FD radiation chemical potential is stationary when its temperature equals the bathT; NC scattering does not enforce zero neutrino chemical potential.

Returned moments, at a fixed incidentE, are: blocked out-event rate, blocked(1−c) angular transport rate, signed/absoluteomega event moments, gain, loss andC(E). Event rates are s^-1 and energy-transfer moments MeVs^-1 **per occupied incident neutrino**; gain/lossC include the prescribed incoming and final occupations. The angular moment is a diagnostic, not an anisotropic inelastic closure. A conditional energy-transfer moment can be nonzero even in LTE; global energy exchange requires an additional incoming-energy/DOS integration and is not computed here.

## Experiments and numerical coordinates

Use two previously qualified neutron baths:

- Warm:T=2MeV,n=.002fm^-3,E=10MeV; radiationFD(Tnu=3MeV,munu=4MeV).
- More degenerate:T=.5MeV,n=.005fm^-3,E=20MeV; nonthermalf(E)=.15FD(E;4,5)+.45exp[−((E−14)/3)²], which stays in(0,1).
- Near-LTE uses the warm bath withFD(Tnu=2.00002MeV,munu=6MeV), a fractional temperature perturbation1e-5.

`adaptive` integrates outgoing energy with QUADPACK vector quadrature. Panels resolveE'=E and thermal/Fermi/recoil redistribution scales; these are numerical panels, not widths or support modifications. The inner angular integral uses eitherq∈[|E−E'|,E+E'] withdc=q dq/(EE'), or an independent direct-cosine rule. Exactq=0 is distributional and is not sampled by interior nodes. Orders48→96 and tail50T→70T are compared, followed by separate angular-only and tail-only changes. The nominal upper cutoff ismax(thermal tail,24redistribution widths) aboveE; nominal tail changes can therefore leave a dense-bath cutoff unchanged. A separate test **actually extends the absolute cutoff by1.5**, from150→225MeV for warm/nearLTE and126.392→189.588MeV for the degenerate bath.

Two sampled-energy representations are compared to that continuum reference: uniform midpoint32/64/128, and redistribution-resolved piecewise Gauss8/16/24/32/48. All weights are positive physicaldE' weights. The latter is a collocation/quadrature representation, not the piecewise-constant radiation-bin ansatz of the separate thermal box. Angular rules96/192/384 are independently refined on the resolving grids; energy and angular errors can alias, so monotone improvement from changing only one order is not assumed.

## Selected converged moments

| Case | Blocked out [s^-1] | Transport [s^-1] | Signed energy into bath [MeVs^-1] | C(E) [s^-1] |
|---|---:|---:|---:|---:|
| warmFD |84986.0606501|87372.9475530|−4763.71847914|190.005490414|
| degenerate packet |243102.431479|262912.423623|354315.155393|−960.464009620|
| nearLTE |84701.0830198|86990.4476168|−5842.66160847|.00348242116071|

The adaptiveq48/tail50 andq96/tail70 results agree within5.8e-14 relative across all seven nonzero moments. Independent cosine96 agrees within1.58e-7, with its largest discrepancy in the absoluteomega diagnostic. Separate actual cutoff-extension contributions are negligible relative to the accepted full moments; raw finite logarithmic tail samples are retained. This is sampled finite-cutoff evidence, not a rigorous infinite-tail bound or a relative-accuracy certificate for exponentially tiny tails. Ordinary underflow in such tails is not interpreted as physical support ending.

Resolved24/q96 matches all seven continuum moments within2.78e-7. The later32/q192,48/q192 and48/q384 experiments test the coupled angular/energy resolution separately. At48/q384, largest errors are4.69e-10(warm),3.98e-8(degenerate),4.71e-10(nearLTE). The unchanged acceptance gate is3e-5. These are actual continuum-rate error measurements at the stated incident points, rather than structural conservation tests.

## Coarse-grid failures and cancellation

**Sixty-nine coarse-grid adequacy gates fail and remain failed in `results.json`.** Uniform128 still has maximum errors17.62%(warm),0.4198%(degenerate) and17.73%(nearLTE). Uniform32 is much worse. Resolved8 also fails selected degenerate moments. Such grids could satisfy a reversible-grid number ledger while resolving these physical rates poorly; algebraic identities do not replace the direct integral comparisons. Reference/final-resolution acceptance is recorded separately from each candidate grid's adequacy. No threshold was loosened to hide those failures.

NearLTE gain/loss are approximately10096.7979378 and10096.7944554s^-1, leavingC=.00348242116s^-1. The implementation evaluates the small difference asloss*expm1(delta), withdelta=omega(1/Tnu−1/T) for FD radiation; for the packet it uses the exact occupation-log ratio. This changes numerical cancellation only. It neither replaces the response nor forces a nonthermal distribution to equilibrium.

A separate outgoing-energy adaptive integral evaluates the **actual reverse kernel** with theE'²/E² measure ratio, without constructing it from KMS. Its nearLTE net agrees with the cancellation-stable result to1.31e-9 relative. At exactLTE(Tnu=T,munu=6), those direct gain-minus-loss kernels give a gross-scaled integrated residual about1.95e-16. The stable FD expression returns zero there by the identity; that algebraic zero is not presented as an independent accuracy test.

## Evidence and limits

`study.py` supplies66 accepted reference/final-grid checks and separately retains candidate-grid errors and69failures. `refine.py` adds73passing separate-control/direct-reverse checks; `refine_dense_grid.py` adds63 coupled-resolution checks; `tail_extension.py` adds21 real-cutoff checks; `check_controls.py` adds25 finite/domain rejection checks. Total accepted checks248, plus the preserved69failed candidate-grid gates. Numerical studies cost approximately2.08+3.22+1.77+.25s. Each completed phase/grid is checkpointed, with105s experiment guards; no runtime stop occurred. This does not rerun or relabel the existing189NC author checks.

The interface is qualified only at these sampled states/occupations/incident energies and floating-point ranges. Arbitrarily strong degeneracy, tiny perturbations, deep-tail relative accuracy, general non-LTE partner spectra and a uniform parameter-domain bound remain open. The next implementation gate is continuum **double-energy** number/energy exchange and energy-grid leakage control, followed by bounded radiation evolution and a consistent matter-energy ledger. No evolving-material, hydro, thermalization-time or interacting-response claim follows from this selected rate qualification.
