# Reviewed CPU primitives bound into controlled free npe matter

This research binding uses the actual NMB source package at `ffb5cc72949f2e5401251847aef4c9c7a98dde78`, copied into this leaf, plus exactly the reviewed scalar CPU lepton and particle-baryon modules. The initial71 baseline blobs are verified, both overlays are SHA-pinned, package origins are checked, and the original mirrors/production files remain unchanged. No package stub, installation or dependency change is used. This is a research composition proof, not a third proposed production API or a new EOS model. Independent review is pending.

Physical constants are explicitly supplied: mn939.56542052,mp938.27208816,me0.51099895069MeV and HBARC197.3269804MeVfm. Thus NMB's slightly different default bare masses are never silently substituted into a weak-source comparison. State is `(nB,T,Ye)` with nn=(1−Ye)nB,np=Ye*nB, electron NET number=Ye*nB. Electrons and positrons are an ideal pair-equilibrated flavor with opposite chemical potentials. **Muon species are absent**, rather than calling a zero-net muon factory that would retain pairs. Baryons are particle-only, constant-mass ideal FD gases; existing forward reference states also retain equilibrium-antibaryon diagnostics. No interacting, density-dependent mass or high-temperature species-complete EOS is supplied.

The optional photon closure is explicit: either absent, or two-polarization ideal trapped LTE photons with eγ=π²T⁴/(15 HBARC³),Pγ=eγ/3,sγ=4eγ/(3T),Cvγ=4eγ/T. Internal rapid EM pair rethermalization and this photon choice are mathematical material closures. They do not constitute a complete EM plasma, photon transport/collision model, or a guarantee that physical equilibration is fast. The cycle uses photons on; the instantaneous source diagnostic reports both closures. Photon energy never escapes here.

## Thermodynamic definitions and initial forward gate

Densities n,s,Cv use fm^-3 (kB=1); e,P use MeVfm^-3; T and rest-inclusive mu use MeV. All responses hold the supplied masses fixed. With hi=mui−T*(∂mui/∂T)|ni, where electron ni is NET number,

- eT=Cv at fixed nB,Ye;
- eY=nB(hp−hn+he);
- en=(1−Ye)hn+Ye(hp+he).

Photon terms enter total e/P/s/Cv; their explicit derivatives with respect to nB,Ye at fixed T vanish. Rest energy is included once in e and in the species chemical potentials. No extra mass-difference Q is added to either the EOS or weak energy source.

Four predeclared baths are `(T,nB,Ye)`=(2,1e−7,.3),(1,1e−7,.3850567101958),(3,1e−7,.5490065797957709),(2,1e−6,.2786277578701). Actual prior accepted free-npe species states are computed, never a zero-net muon completion. The116 selected forward/root/response comparisons pass; maximum relative discrepancy7.14e−14. Only after this gate,56 checks compare T/Ye/nB finite differences, entropy derivative, Legendre and corrected enthalpy identities, and an independently integrated Planck spectrum. Maximum finite-difference discrepancy1.49e−7 against the predeclared3e−5 gate. They exercise both photon closures and the actual energy/composition derivatives. Such numerical agreement is not physical model uncertainty or AD qualification.

## Prescribed-density weak-off cycle

nB(t)=n0 exp[ln2 sin(2πt/τ)],n0=1e−7fm^-3,T0=2MeV,Ye=.3,τ=1s. No momentum/metric/hydro equation is solved; mechanics is prescribed externally. The first law is d(e/n)/dt=P*ndot/n², giving

`Tdot = {[(e+P)/n−en] ndot}/Cv`.

Classical RK4 at32 and64steps evaluates the actual EOS on each stage. All18 cycle gates pass, with full partial checkpoints retained. A different constraint calculation solves s(T,n,Ye)/n=constant directly at six times; fresh old-EOS forward evaluations at those states and tighter/tail100 native quadrature at the compressed state provide37 additional checks including32→64 path refinement. There is no EOS interpolation or fitted heat capacity.

The64-step temperature range is1.5815874774–2.5297599619MeV. Max entropy-per-baryon drift1.64e−8; independent entropy-root temperature discrepancy8.12e−9; return-temperature error1.69e−9;32→64 T-path discrepancy1.43e−7. Pressure-work uses a separate composite Simpson integration on saved EOS endpoint states, not the RK4 temperature update or a telescoping energy identity. Net work3.34e−9MeV/b and actual final-minus-initial specific energy−2.58e−7MeV/b are both small; their mismatch divided by **gross |Pdn/n²| work40.544MeV/b** is6.45e−9. It is never divided by baryon rest energy or a near-zero net work. Separate quarter-cycle work/energy errors range5.56e−7–1.35e−6; these honest nonzero local quadrature errors are recorded alongside the global gate. These are selected timestep/path checks, not a general convergence-order or long-time claim.

`density_cycle.png/.pdf` show the actual prescribed density, RK4 temperatures versus independent entropy roots, and entropy residuals. Render inspected: labels/data readable;32/64 curves overlap in T but separate clearly in the entropy panel.

## One fresh weak point, no weak-on evolution

At T2,nB1e−7,Ye.3, the newly bound species' **actual total chemical potentials** are supplied directly to accepted full-current physical EC, positron-capture and beta event integrals. The frozen worker's physical source functions are imported under unique names; its legacy state factory is not called. Each profile runs in a fresh `-I -B` process with source hashes and loaded dependencies recorded. No saved moments are used as feedback. The incoming νe and antinue occupations are empty. Both complete profiles and independent event coordinates pass: the six positive number/energy moment refinements are≤5.22e−9, and the prior accepted same-bath absolute moments are reproduced. Source calls took2.11 and5.83s.

Fine-profile escaping number sources (fm^-3s^-1): REC3.3088603482e−7,RPC1.1173192812e−6,Rbeta4.2587395066e−11. Thus R=Rbeta+RPC−REC=7.8647583379e−7, Q=sum escaped neutrino/antineutrino energies=1.5155020638e−5MeVfm^-3s^-1. Direct species-energy and material-loss checks use the actual kinematic moments. Le, baryon and charge ledgers are checked independently of the total-energy sum.

For the locked instantaneous ndot2.5e−8fm^-3s^-1,

`Yedot=R/n`,

`Cv*Tdot=[(e+P)/n−en]ndot−Q−eY*R/n`.

With photons, Yedot7.8647583379s^-1,Tdot−1.7284560792MeV/s. Without photons, on the same material/weak-source point and imposed ndot,Tdot−2.7824996835MeV/s. Cv is7.6672521652e−6 versus4.9269462117e−6fm^-3. This is an instantaneous closure sensitivity only. Actual pair densities are differentiated in T,n,Ye and their compression/weak sources removed: inferred internal EM creation is2.1658207596e−7 and2.1658207594e−7fm^-3s^-1 from e−/e+ separately. It changes pair populations equally, preserving net charge/Le. First-law residuals are also explicitly divided by gross escaping Q after subtracting mechanics/composition; rest energy does not hide weak-source errors.

## Qualification, preserved failures and remaining gates

Current267 selected checks pass:116 forward,56 partial/Planck,18 cycles,37 cycle/old-EOS/tail,14 fresh weak-point,26 aggregation/admission/source-refinement. These are author checks, not independent acceptance. No physical current/model, tolerance or predeclared scientific gate was retuned.

Three initial implementation/metadata errors are preserved: (1) NumPy integer JSON count reporter failure before first forward report, fixed by native scalar reporting; (2) an independent check mistakenly used `Ts−T sum n muT` instead of the correct `Ts+T sum n muT` for `e+P−sum n h`, causing8 check failures despite48 other partial checks passing; its exact source/results remain before the algebra correction; (3) a literal prelock timestamp03:35 was a transcription error, corrected to original lock filesystemmtime03:33:53.593683UTC, before the first scientific call. Exact original LOCK and timestamp note remain. States/gates/protocol were unchanged. The figure tool emitted existing font-cache warnings and used an authorized temporary cache; no dependency was installed.

The research admission T.5–5MeV,nB4e−8–2e−6fm^-3,.05<Ye<.9 is a bounded request guard, not a uniform error certificate. No weak-on trajectory, full neutrino inventory, transparency equilibrium, hydro, full EM kinetics, mean field, antibaryon completion, empirical uncertainty or AD/Hessian fidelity is claimed. Different independent review must precede accepted integration/publication. Minimum-runtime/full CI limitations of the two CPU candidates remain separate. Existing production APIs and frozen sources are unchanged.
