# Independent numerical review: free neutral-current scattering

15 October 2026. **Selected valid physical/numerical cases pass 15 independent checks. General input-contract acceptance is blocked by four preserved counterexamples pending repair.** `probe_neutral_current.py` generates the evidence. Initial frozen-source failures are preserved in `initial_neutral_current_review.json`; the same cases with an additional outer-q refinement are preserved in `neutral_current_refined_initial.json`.

## Independent moments and absolute normalization

This review integrates in **momentum transfer q and nucleon energy transfer omega**, rather than the author's fixed-cosine/outgoing-energy coordinates. For a massless neutrino,

```
Eout=Ein-omega,
-q <= omega <= min(q,2Ein-q),
dc = q/(Ein Eout) dq .
```

Adaptive quadrature explicitly resolves recoil and thermal/Fermi support. At the new finite-mass state T=3 MeV, n=3e-4 fm^-3, Ein=7 MeV, independent total, angular-transport and bath-energy moments agree with the original implementation within 9e-16 relative. The bath-energy moment is negative at this state, correctly retaining upscattering/energy extraction from the bath rather than clipping it to zero.

A held-out heavy common-mass point, M=1e7 MeV,T=1.7 MeV,n=3e-6 fm^-3,Ein=13 MeV, reproduces absolute neutron static total and transport cross sections within 2.82e-6 and 3.74e-6, respectively. These are finite-M approach errors under the model, not a fitted rescaling. Additional outer-q panels around the lepton kinematic boundary q=2Ein change the integrated heavy rates by less than 6e-15 relative; the signed energy moment changes by 2.1e-13. Thus an unresolved narrow q-boundary layer is not driving the elastic-limit comparison.

The prefactor contains GF² but no charged-current CKM factor. The final Eout² measure is consistent with the one-helicity neutrino DOS and the spin-two free baryon response. In the free unpolarized leading-current model, density and each Pauli-spin component have the same response; that equality must not be carried into interacting/spin-correlated matter without a new response model.

## Reciprocity and closed-grid structure

Three new transfer pairs verify the reverse-kernel relation including the **DOS ratio**:

```
log K(Eout→Ein)-log K(Ein→Eout)
 = 2log(Ein/Eout)-(Ein-Eout)/T .
```

The energy-squared ratio cannot be dropped when constructing occupation-number in-scattering.

On a new four-energy irregular weighted grid, this review reconstructs all unordered forward/reverse event pairs using adaptive angular integrals, independently of the author's transition matrix. Depositing equal/opposite event number changes reproduces the reported bin sources. An FD spectrum at the matter temperature and arbitrary chosen neutrino chemical potential is stationary; a nonequilibrium spectrum conserves total neutrino number. Summing each event's bath energy transfer reproduces the negative radiation-energy source.

These are structural closed-grid identities. They do not establish continuum redistribution rates, outgoing-energy truncation accuracy for a radiation evolution, a transport closure or a relaxation time for a physical broadband spectrum.

## Preserved input-contract failures

All four calls below are accepted by the initial frozen implementation:

- `moments(species,7,angle_order=8,tail=-1)` returns rates instead of rejecting its negative integration-tail control.
- `moments(species,7,angle_order=8,radiation=(-1,0))` evaluates a negative radiation temperature instead of rejecting it.
- `discrete_collision` accepts a NaN in the occupation array and returns NaN collision/source data.
- `discrete_collision` accepts a NaN in a quadrature-weight array and returns NaN collision/source data.

The source tests compare positivity/ranges but omit finite checks; comparisons involving NaN are false. Input validation must establish finite positive integration controls, finite radiation temperature/chemical potential with Tnu>0, and finite energy/weight/occupation arrays before computing. These defects do not invalidate the passing selected valid-control physics results, but they prevent acceptance of the advertised general numerical calling contract until independently repaired.

The parent has commissioned a scoped leaf repair while preserving initial source and results. This review does not edit author code. A distinct repair verification will supersede only the resolved failure-contract findings.

All model boundaries remain: free ideal NR baryons, leading vector/axial vertices, no recoil-current terms, weak magnetism, correlations, RPA, antiparticles or GRMHD closure. No physical external finite-T code comparison is claimed by these numerical checks.
