# Static screened-ion Lorentz gas: physical electron transport reference

15 October2026. **Author-frozen research prototype; independent review required. 129/129 checks pass.** This is a controlled, unmagnetized electron gas scattering from immobile, uncorrelated point charges. It derives the conductivity collision time from a screened Born/Mott differential cross section; it does not insert an adjustable relaxation-time law or Coulomb logarithm. It is not a realistic correlated crust, core, mobile plasma or whole-star conduction prescription.

The existing NMB conduction qualifications remain intact: crust/envelope prescriptions retain their source/domain conventions, while `core_conduct.py` is a lepton-core reference including its declared proton-pairing treatment; baryon conduction and changed screening remain separate options. This new leaf edits neither module and supersedes neither physical model.

## Primary anchors and deliberate model

[Potekhin, Baiko, Haensel & Yakovlev1999, Eqs.3,6–9](https://arxiv.org/html/astro-ph/9903127) provides the screening susceptibility, Born transport-frequency normalization and relativistic angular factor. [Potekhin1999, Eqs.9,16–19](https://arxiv.org/html/astro-ph/9909100) gives screening/energy averaging and thermoelectric conventions. The [Geant4 electron screened-scattering reference](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/elastic_scattering/elecnuc.html) is an additional primary implementation reference for separating screening and Mott factors; its atomic screening/non-Born prescription is **not** imported here. The equations below are derived for our simpler named ansatz.

Electrons are ideal relativistic spin2 particles, E=sqrt(p²+me²), with physical me=.51099895069MeV, total mu including rest mass, and FD occupation per spin. The ion charge is Ze with ni=ne/Z. Ion positions are quenched uncorrelated external scatterers, S(q)=1; the background absorbs momentum without recoil or energy exchange. Ions are not evolved and do not supply a heat capacity, diffusion current or ion screening. There are no electron-electron collisions, correlations, phonons, magnetic fields, nuclear-size factors, band structure or localization corrections.

The potential is the **local static susceptibility/Yukawa** model `U(q)=4pi Z alpha/(q²+ks²)`, with `ks²=4pi alpha (dn_e/dmu)_T`. This uses the actual finite-T electron susceptibility, not a zero-T replacement. It is a long-wavelength screening ansatz extended over the scattering q integral, not the full q-dependent Lindhard/Jancovici dielectric response. The electron-only density is the particle density; an equilibrium positron diagnostic rejects npositron/ne>1e−8 and retains finite log ratios/ordinary representation flags. Positrons contribute neither EOS nor screening nor transport in this declared approximation. Actual selected pair ratios are much smaller. The generic relativistic particle EOS is reused with **electron mass**, not baryon masses or rest conventions.

Born and semiclassical approximations are physical model choices, not numerical error certificates. Zalpha/v varies with electron energy and grows at small v; a Fermi-velocity diagnostic is recorded but is not an error bound and is not representative of a nondegenerate gas. No claim about the accuracy of omitted non-Born effects follows from numerical agreement. A local FD state is prescribed by reservoirs/slow thermalization: the Lorentz collision operator is elastic and does not itself relax an arbitrary isotropic distribution to FD.

## Cross section and the exact l=1 eigenvalue

Use natural hbar=c=kB=1 and alpha=1/137.035999084. For scattering cosine c and x=sin²(theta/2)=(1−c)/2,

`dσ/dΩ=4 Z² alpha² E² [1−v² x]/(4p²x+ks²)²`, `v=p/E`.

This is the screened **Born** Dirac/Mott factor, not the all-orders point-Coulomb Mott series. The numerically stable code writes E²[1−v²x]=me²+p²(1−x), preserving backward suppression without cancellation. Independently normalized external Dirac spinors reproduce the spin factor and absolute potential prefactor: `dσ/dΩ=E² |U(q)|²/(4pi²) * (1/2 sum_spins |u'†u|²)/(4E²)`. The nonrelativistic unscreened limit is Rutherford. No target-spin factor is inserted for a static charge; the electron initial spin is averaged in dσ, while spin2 is counted in the electron phase density/moments.

For an elastic isotropic kernel, the spherical harmonics are collision eigenfunctions. The driving charge/heat perturbation is l=1, whose **derived** frequency is

`nu1(E)=ni v integral dΩ (1−cos(theta)) dσ/dΩ`.

It is not the total scattering frequency. Exact FD final-blocking factors cancel in the elastic gain-minus-loss operator: f'(1−f)−f(1−f')=f'−f. Thus no extra final1−f multiplies nu1; degeneracy enters through the transport derivative−df/dE.

Let a=ks²/(4p²), b=v². Direct angular integration gives

`nu1=2pi ni Z²alpha²/(p²v) I(a,b)`,

`I=integral_0^1 dx x(1−b x)/(x+a)²`,

`I=(1+2ab)ln(1+1/a)−1/(1+a)−b[1+a/(1+a)]`.

The source's conventional Coulomb logarithm is Lambda=I/2, reproducing its4pi prefactor. For strong screening a>=4 the code uses the convergent exact expansion `I=a^-2 sum_k>=0 (−1/a)^k(k+1)[1/(k+2)−b/(k+3)]`, not subtracting nearly equal large terms. The frequency's strong-screening branch also combines p²/v=pE before evaluation. Unrepresented positive integrals/frequencies raise rather than becoming physical zeros. There is no arbitrary minimum impact parameter or physical angular cutoff.

A direct gain-minus-loss angular integral of a dipole oriented at a nontrivial angle independently reproduces−nu1 times that dipole. The l=1 collision time tau=1/nu1 is therefore exact **within this static isotropic Boltzmann kernel**, rather than an arbitrary surrogate collision model. Realistic plasma corrections remain excluded.

## Thermoelectric moments and charge convention

The spin2 phase measure and1/3 angular projection give

`Kn = g/(6pi²) integral_0^infinity dp p² v² tau(E)(E−mu)^n [−df/dE]`, `g=2`, n=0,1,2.

Natural charge is **Heaviside-Lorentz**: eHL²=4pi alpha, with electron charge q_e=−eHL. Literature Gaussian eG²=alpha must not be mixed numerically with this conductivity convention. Screening is ks²=eHL² chi=4pi alpha chi in either physical description. Define electrochemical field `Estar=E−grad(mu)/q_e=E+grad(mu)/eHL`. The current and heat flux are

`j=q_e²K0 Estar−q_e K1 grad(T)/T`,

`JQ=q_e K1 Estar−K2 grad(T)/T`.

Our Seebeck convention is **Estar=S grad(T) at j=0**, so

`sigma=eHL²K0`, `S=K1/(q_e T K0)=−K1/(eHL T K0)`,

`kappa_open=[K2−K1²/K0]/T`, `Peltier=T S`.

The open-circuit subtraction matters. K2/T is the heat coefficient at Estar=0, not open-circuit conductivity. With forces XN=(q_e E−grad(mu))/T and XT=−grad(T)/T², the particle/heat Onsager matrix is T[[K0,K1],[K1,K2]]. Its positive covariance follows from the positive integral weight; numerical checks preserve that determinant and positive entropy production. The explicit convention prevents an ambiguous thermopower sign.

Outputs use natural units only: K0[MeV],K1[MeV²],K2[MeV³],sigma[MeV in HL charge units],kappa[MeV²],S dimensionless with kB=1. nu1 is MeV and tau is MeV^-1. Input density fm^-3 is converted by(hbarc)³ using197.3269804MeVfm; no guessed CGS conductivity conversion is provided.

## Selected measured results and numerical evidence

| T [MeV] | ne [fm^-3] | Z | ks² [MeV²] | sigma [MeV,HL] | kappa_open [MeV²] | S |
|---:|---:|---:|---:|---:|---:|---:|
|.003|1e−14|1|2.340420604e−6|.09005708119|.01143907764|−28.02105414|
|.003|1e−11|1|.0006249188514|.7123237290|.07879028309|−4.808288279|
|.07|1e−7|2|.07564722608|91.65271219|231.0760653|−.981245076|
|.25|1e−4|6|7.459134536|315.9840798|2835.608858|−.337893495|

The first state is nondegenerate; its zero-Estar coefficientK2/T=.2235719929 differs strongly from the open-circuit value. The table is a formal Born/Lorentz-model reference, not a calibration of real stellar matter. Pair log ratios span roughly−331 to−55 in these selected states. All finite-T roots, screening, Kn and controls are retained in results.json.

`validate.py` has98PASS in.20s. Analytic I agrees with a separately transformed positive angular quadrature over a=1e−8...1e8 and b=0,.37,1 within7.93e−14. Direct angular cross-section integration matches nu1 within6.67e−15. Original-momentum transport moments independently reproduce the energy-window quadrature within1.58e−14. A separate **nested angular-cross-section plus momentum** integral reproduces all3 moments, checking the whole chain rather than only substituting the analytic tau. Tail60→85 and tightened tolerances leave selected moments within8.22e−15. Independent density integrals gate forward roots before finite-difference screening checks; two decreasing fixed-mu steps agree with the actual susceptibility within1.64e−7.

At fixed ne=1e−7fm^-3 and Z1, the re-solved T/TF=.02,.01,.005 sequence approaches Wiedemann–Franz, `kappa/(sigma T)->pi²/(3eHL²)`. Relative errors are.00181022,.000453466,.000113424, approximately T². The final Fermi-surface Drude relative correction is6.58e−5, with sigma0=eHL²ne tauF/EF. The finite-T Lorenz ratio was computed from the full moments, not imposed. This numerical limit does not make WF exact away from degeneracy.

`check_physics.py` adds31PASS: external-spinor Mott normalization, oriented l1 angular collision operator, exact elastic-Pauli cancellation, positive Onsager entropy, and a new T.052/ne2.4e−7/Z1↔3 family. At fixedne neutrality makes nu proportional to Z (ni=ne/Z), sigma/kappa proportional to1/Z, while screening and S remain unchanged; the held-out scaling passes. Pair-rich, nonfinite/zero controls, invalid ion charge and unrepresented angular integrals reject explicitly. No numerical failures arose in this batch. No dependency installation, production, paper or git modification was made.

## Qualification and next gates

This is a reproducible **static uncorrelated external-ion** collision/conductivity slice, with selected finite-T roots and low-T limits. It does not estimate real crust/core model uncertainty or qualify an arbitrary state rectangle. It supplies neither correlated-ion S(q), lattice/phonon/impurity-distribution physics, dynamic screening, realistic non-Born corrections, electron-electron heat transport, pairs, magnetized tensors nor baryon conduction. Treating dilute lattice impurities would require charge-variance/density and host-lattice conventions; substituting this neutral single-ion Z model for a realistic Qimp formula is not justified.

Independent review should check the Born normalization/charge units, l1 factor, actual susceptibility and open-circuit sign/subtraction, then reproduce a new state using independent angular/energy coordinates. Next useful physical extensions are prescribed S(q) with transparent limits, finite-q dielectric screening, electron-electron collisions, and a fixed-background thermal diffusion demonstration with explicit units/boundary reservoirs. Existing NMB conductivity kernels and their separate validity qualifications remain available throughout. No width fit, phenomenological collision time or performance certificate replaces these physical gates.
