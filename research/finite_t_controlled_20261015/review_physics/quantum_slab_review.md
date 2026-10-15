# Independent physics review: irradiated FD-nucleon spectral slab

15 October 2026. Reviewed `quantum_slab/fixed_bath.py`, `guarded_capture.py`, research note, main 506-check results, separate eight-check guard report and preserved initial failures. Reviewer owns only new `review_physics/quantum_slab_*` files. No author/frozen module was changed. Quota/account/window matched at start (4% used).

**Accept the twelve selected-energy fixed-bath transfer cases and the leaf-local collapsed-panel guard at its documented witness state.** The combination contains the three named electron-channel branches with their inverses, matched FD baryon/lepton inputs, and correct signed bath energy/lepton exchange. No blocking physics inconsistency found. This is not a converged energy-integrated spectrum/luminosity, a material time evolution, a relativistic/GM1 weak response, or universal qualification of the fallback guard.

## Channel inventory and factory affinity

The EOS factory imposes net electron density np and independent free-FD neutron/proton densities with common M and constant offsets `(Q,0)`. Capture uses the factory's absorption-oriented PairState; crossed decay explicitly reconstructs its spontaneous n→p state. Electron neutrinos receive electron capture/inverse absorption. Antineutrinos receive positron capture/inverse absorption **plus** neutron beta decay/three-body inverse, each calculated before the positive logarithmic sum.

These named channels share `mu_nue=mu_e+mu_p−mu_n`, `mu_antinue=−mu_nue`. Their KMS relation is appropriate to the actual EOS occupations, not a substituted classical nucleon ratio. The earlier independent factory/free-energy and finite-M crossed/capture checks support this matching. There is no classical baryon energy substitution and no claim that supplying this factory alone supplies interactions or missing weak-current terms.

The per-channel KMS checks and separate quadrature refinement are both needed: reciprocity can hold even when both directions suffer a common quadrature error. The note makes this distinction correctly. The raw table has twelve energy/flavor/state combinations, not twelve energy groups of a qualified total-spectrum integration.

## Ray, angular measure and two-face factors

For the one-helicity density measure `D E² dE`, the axial angular measure is `dmu/2`. The exact positive ray solution is a sum of bath emission and attenuated incoming occupation. It preserves occupation bounds without clipping and avoids opaque-limit cancellation for tiny f_eq.

Writing `A=c D E²*1e39`, `B=1/2−E3(tau)`, the left outgoing magnitude is `A/2[f_eq B+f_right E3]`, with the analogous right-face expression using f_left. Incoming magnitudes are `A f_left/4` and `A f_right/4`. Therefore

`F_gain=A[f_eq−(f_left+f_right)/2]B`.

These are the implemented factors. The cgs density conversion, one neutrino helicity and two-face normalization are correct. In the thin limit the gain is the depth integral of the **net** isotropically averaged collision source, not spontaneous emission alone under irradiation. In the thick active-mode limit the summed outgoing magnitude tends to `A f_eq/2`, while net gain still subtracts both imposed incoming hemispheres. Thus thick radiation gain need not be positive.

Attenuation uses `(j+chi)/c`, consistent with the occupation collision law; `chi/c` remains the true absorption coefficient. A strong emission/low-vacancy example can have relaxation optical depth much larger than true absorption depth, as the note illustrates. This does not create a new scattering process.

For extremely tiny tau, `1/2−E3(tau)` itself can eventually lose floating-point precision even though net gain is assembled stably. The tested ladder begins at 1e-6. A wider ultra-thin qualification should use a controlled small-tau evaluation; acceptance does not cover arbitrarily small optical depths.

## Signed material exchange and units

The bath energy counterterm is `−E F_gain`; electron-lepton radiation gain is `+F_gain` for nue and `−F_gain` for antinue, with the bath opposite. For antineutrino absorption the radiation number decreases, its electron-lepton charge increases, and the bath electron-lepton source is negative; this is the correct composition direction.

All these source entries are **column-integrated per-area per-energy exchanges** inherited from the flux calculation, not local volumetric heating or a per-baryon composition rate. If a later homogeneous material evolution is added, integrate over energy and divide the lepton source by the baryon column `nB*1e39*L` for dot(Ye); local deposition profiles require spatial collision moments. The generic bath-number counterterm is merely paired bookkeeping and is not a claim of conserved neutrino particle number in matter.

The bath is externally held at fixed T/composition, so returning the opposite source diagnoses its reservoir exchange. It does not integrate a closed-system material EOS energy/entropy equation. Different prescribed incoming hemispheres further supply boundary energy/lepton reservoirs.

## Collapsed-panel numerical guard

The preserved witness has distinct raw electron-energy edges separated by about one floating-point spacing, but identical transformed `sqrt(E−me)` endpoints. Skipping this **unrepresentable transformed-width** panel is a numerical-domain action, not threshold smoothing. The algebraically equivalent momentum `p=x sqrt(2me+x²)` avoids cancellation in the mapped threshold coordinate.

An independent estimate in `quantum_slab_probes.py` evaluates each discarded raw-E interval's smooth integrand, including its actual angle weight, prefactor and FD/response factors. At the documented T=2 MeV, nn=.002, np=.001, nue E=60 MeV witness, one panel is discarded. The estimated omitted fractions of j and chi are both **4.17e-18**, far below the selected-state quadrature budget. This is a selected smooth-integrand estimate, not a universal rigorous error bound for arbitrary exceptionally narrow responses.

The wrapper records the original failure, requires a discarded-panel witness and leaves frozen source unchanged. For a broader API, validate all quadrature/state inputs and distinguish the specific collapsed-panel exception more narrowly than arbitrary ValueError; requiring any discarded panel is not by itself a proof that every coincident error had that cause. No masking problem was found in the recorded valid-input witness. The guard is accepted locally, not as a universal error-recovery policy.

## Additional independent transfer probes

`quantum_slab_probes.py` writes `quantum_slab_probe_results.json`; all **seven checks pass**:

- Six direct integrations of `j(1−f)−chi*f` over signed angular directions and optical depth, using an intentionally **absorbing** synthetic bath with unequal incident hemispheres, reproduce the analytic radiation gain for both flavors at tau=.03,1,20. Largest relative discrepancy **8.90e-11**. Energy deposition and flavor-dependent matter lepton signs are checked explicitly. This uses the literal collision law rather than the author's already reduced E2 depth source.
- The discarded-panel estimate above is below its 1e-12 selected-state gate.

The main 506 checks and separate eight guard checks additionally provide coefficient refinement, boundary angular integration, depth source, ray ODE, blackbody and heavy-static normalization evidence. Their physical model remains shared with the underlying response/current kernels. The positive flux repair and stricter small-reference absolute error control resolve preserved numerical failures without changing physical prefactors.

## Next gate

Extend to a state-specific converged neutrino-energy grid only after guarding threshold panels, both chemical-potential signs, narrow Fermi features and thermal tails. Integrate incoming, outgoing and bath exchange separately before comparing a potentially small net difference. Quantify which tail contributions require p~M and remain outside NR physical control. Only then report total luminosity/heating/lepton exchange or attempt an EOS-coupled slab/box evolution. Current recoil, weak magnetism, interactions, scattering and redistribution remain separate qualified extensions.
