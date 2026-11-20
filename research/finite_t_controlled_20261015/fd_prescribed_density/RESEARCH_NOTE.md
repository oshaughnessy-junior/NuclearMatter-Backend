# Prescribed-density FD element: work, entropy and coupled weak-source equations

2026-10-15. **55/55 author weak-off numerical checks pass in76.03 s.** This research leaf extends the matched free-FD element to a prescribed baryon-density history and independently checks the mechanical-work term. It supplies the weak-on equations and a costed numerical design, but **does not claim a weak-on density trajectory**. Independent review is required.

## Controlled state and energy convention

Use the same common-mass free NR n,p plus pair-equilibrium relativistic e± EOS as `fd_fluid_element`, with fixed physical Q neutron offset, charge neutrality, independent Ye and no muons/interactions/nuclei. These offsets are not the composition-specific calibrated offsets of the separate threshold demonstration. Density is prescribed externally; no momentum, gravity, density PDE or hydro solver is supplied. This is a comoving element with fixed baryon number, so changing nB changes its volume.

Write u=ε/nB, including neutron Q and total e± energy. A subtracted common nucleon rest energy is constant per baryon and causes no work/source ambiguity. The optional photon EOS contribution εγ=π²T⁴/[15(ℏc)³] is explicit; the tested density history uses **photons=False**. Derivative checks cover both photon choices.

For Ai=μi−T(∂μi/∂T)ni, with the electron variable being net conserved charge including pair equilibrium,

uT=(Cv,n+Cv,p+Cv,e[+Cv,γ])/nB,
uY=Ap−An+Ae,
un≡(∂u/∂nB)T,Ye = [(1−Ye)An+Ye(Ap+Ae)−u]/nB.

All chemical potentials, temperature derivatives, heat capacities, pressure and entropy are genuine FD values from the same EOS. Q is retained; no3T/2 or classical composition-energy substitution is made. Photons contribute to u,T and pressure but not to the Ai charge derivatives. In particular their contribution to un is−εγ/nB².

The independent thermodynamic form

nB² un=P−T(∂P/∂T)nB,Ye

is checked against the Ai expression. For the particles, PT=s_density+Σni μi,T, where the e± term uses net electron density; photons add4Pγ/T. Direct density differences of both u and entropy supply a separate numerical check, rather than relying only on agreement of two analytic identities.

## Mechanical and weak equations

For a prescribed nB(t), the first law per baryon is

du/dt=(P/nB²) dnB/dt−q,
dYe/dt=F,

where transparent complete charged-current radiation gives

F=(Ranti−Rν)/nB,
q=(Qν+Qanti)/nB,
Rα=∫D E² jα(E)dE,
Qα=∫D E³ jα(E)dE,
D=1/[2π²(ℏc)³] (one helicity).

The complete named inventory is electron capture/inverse for νe and positron capture plus crossed neutron decay/inverses for anti-νe; transparent means radiation occupations are zero. Each new state must recompute the physical sources with its actual nB,T,Ye and unchanged dispersion/current parameters.

The temperature equation is therefore

dT/dt={[(P/nB²)−un] dnB/dt−uY F−q}/uT.

Equivalently, its density term is T PT dnB/dt/nB². The entropy first law is

T ds/dt=−q−(μp+μe−μn)F,

for entropy per baryon. Mechanical work cancels from this entropy balance. With weak processes off, F=q=0, the motion is reversible and s stays constant; compression can raise T even though no weak energy source exists. With weak processes on, neutrino energy loss alone does not determine the sign of dT/dt because both mechanical work and composition energy matter.

## Weak-off numerical experiment

The prescribed cycle is

nB(θ)=10⁻⁴ exp[0.02 sin²(πθ)] fm⁻³,
θ=t/(2×10⁻⁶ s), 0≤θ≤1,
Ye=0.3, T(0)=2 MeV.

It compresses by2.0201% and returns to the starting density. The time label specifies the external history; no hydro consistency or microscopic equilibration-timescale certificate is inferred from it. The warm dilute free-NR domain stays close to the previously qualified state.

One calculation integrates the temperature ODE using the genuine un and uT, with no entropy constraint inserted into that ODE. A separate calculation solves the full FD entropy equation at seventeen densities along the cycle. Their largest temperature discrepancy is6.91×10⁻¹¹ relative; the temperature ODE conserves entropy to2.06×10⁻¹¹ relative. Peak temperature is2.02462272412 MeV at nB=1.02020134003×10⁻⁴ fm⁻³. Entropy is6.01956871769 per baryon.

An additional independent pressure integral computes work along the entropy-root path:

W=∫P/nB² dnB=∫(P/nB)dln nB.

Separate positive compression and negative expansion terms avoid concealing error by cancellation around the full cycle. Four- and eight-node Gauss rules agree within5.40×10⁻¹³ relative. Compression gives+0.0711172401816 MeV per baryon and expansion−0.0711172401816; each agrees with the directly reevaluated full EOS energy change within6.22×10⁻¹³ relative. Final entropy-root temperature returns to2 MeV within1.56×10⁻¹⁵ relative. These work/entropy checks establish one controlled reversible path, not weak-source feedback.

Density finite differences at decreasing relative steps2×10⁻⁴ and10⁻⁴ validate un to6.45×10⁻⁹ relative, including a new warmer/changed-composition state. The entropy Maxwell density derivative agrees to1.42×10⁻⁸; the pressure/un identity agrees to6.22×10⁻¹⁵. Photon-option derivatives pass, while the cycle itself is photon-free.

## Preserved runtime failure and bounded repair

The initial implementation nested adaptive pressure quadrature over bracketed entropy inversions, each calling full FD density roots. It was interrupted near the120 s numerical budget before completing the independent work check; **no completed first-attempt qualification is claimed**. `initial_density_element.py`, `initial_weak_off.py` and `initial_work_budget_stop.txt` preserve it.

The accepted implementation changes numerical algorithms only within this leaf: entropy inversion uses Newton updates with the exact FD derivative ds/dT=uT/T, and work uses explicit Gauss4→8 refinement. No EOS/rate approximation, fitted curve, acceptance-tolerance change or source floor was introduced. An explicit105 s deadline is checked at EOS calls, and phase/partial outputs are saved even on a failure. The completed retry cost76.03 s; it does not erase the earlier interrupted attempt.

## Weak-on design and open gate

A robust next experiment can split each short step into: (1) evaluate the actual complete physical weak moments at the current full state, update u and Ye at fixed nB, and invert the full EOS for T; (2) move to prescribed next nB at fixed new Ye and entropy, independently checking∫P dnB/nB². This is a first-order operator split, requiring at least2/4-step comparison; it is not a higher-order integrator by construction.

Maintain independently assembled per-baryon ledgers

u(t)+Eescaped(t)−Wmechanical(t)=u(0),
Ye(t)+Lescaped(t)=Ye(0).

The mechanical pressure integral, reconstructed EOS energy and charged-current energy moments must each be evaluated independently. Separate positive/negative work and radiation components when testing small net balances. Density, composition and temperature derivatives must use the same fixed offsets/current model. A reused exact initial physical source requires the repaired full-state/source/control/environment qualification fingerprint; changed-state rates must be freshly evaluated.

The source-qualified isochoric study required about94 s for the relevant fresh source/refinement work, while the current independent entropy/work baseline takes76 s. The work cost depends on how often it is evaluated, so a complete density-feedback/refinement experiment has not yet been demonstrated to fit a120 s bound. No weak-on density trajectory is run or accepted in this batch; the next batch should first profile/reuse exact-state EOS evaluations, bound each physical source call, and retain independent work checks. Reusing fixed rates as changing-state feedback is not an alternative.

Inherited open gates remain: the original T=2 strict transfer-wrapper heavy-static integrated failure, general interacting-EOS/current validation, long-time integration accuracy, nonempty radiation/transport closure and full hydro. None is closed by reversible weak-off work agreement.

Reproduce with `python outputs/nmb-finite-t-20261015/fd_prescribed_density/weak_off.py`. Raw derivatives, ODE samples, independent work values, partial checkpoints and source hashes accompany this note.
