# Independent numerical review: crossed neutron-decay optical channel

15 October 2026. **Accept the crossed-channel implementation at the selected test states and held-out heavy-mass limit.** Its finite-electron-mass endpoint map is independently verified. Preserve the separate endpoint limitation in the frozen earlier capture kernel before expanding a combined opacity scan.

This review executes **12 passing checks**, saved in `crossed_review.json`, with `probe_crossed.py` for reproduction. A separate independent reproduction of the earlier capture limitation is in `frozen_endpoint_independent.json` and `probe_frozen_endpoint_independent.py`. No author files were modified.

## Crossing, endpoint and support

Spontaneous decay has ordered baryons n→p, transfer `omega=-E_nu-E_e` into baryons and momentum-transfer magnitude `sqrt(E_nu²+p_e²+2E_nu p_e cos(theta))`. Reversing baryons and omega gives the inverse three-body absorption. Combining the electron vacancy/occupation ratio with the baryonic KMS relation yields `log(j/chi)=(mu_n-mu_p-mu_e-E_nu)/T`, matching the code.

The true electron endpoint contains `p_e=sqrt(E_e²-m_e²)`. Mapping `E_e=m_e+x²` regularizes its integrable square-root corner without changing reaction support. This review independently integrates in **electron momentum p_e**, where `E_e p_e dE_e=p_e² dp_e`, using adaptive scalar integration and a distinct high-order angular rule. No author optical energy-panel generator is reused.

That momentum integral agrees with the crossed coefficients at the author's degenerate point `(T,nu_n,nu_p,mu_e)=(3,20,5,5)` MeV, E_nu=1 MeV, and a new modest point `(2,8,1,1.3)` MeV, E_nu=0.8 MeV. Energy/angle order 24→48 and tail 40→70 tests pass. KMS is retained, and finite thermal log-rates are not replaced by the zero-temperature support test.

## Integrated heavy-mass normalization

A held-out dilute state uses T=1.5 MeV, mu_e=1.1 MeV, `(nn,np)=(8e-10,2e-10)` fm^-3 and common kinetic M=1e7 MeV. Independent outer neutrino integration compares both number and energy moments with the previously qualified static neutron-decay target integrals. Both normalize within 1e-5, and outer-order refinement passes 2e-6. The remaining finite-M discrepancy is not relabeled as quadrature error or corrected by a fitted normalization.

This checks the optical prefactor against the restricted static limit in another state. It is not an independent derivation of the full physical weak current. The allowed leading-current/nonrelativistic approximation still excludes recoil-current corrections, weak magnetism, relativistic support and interacting responses.

## Preserved failures and old capture limitation

The author correctly preserved two distinct initial failures: inadequate absolute tolerance in a suppressed independent tail integral, and a true degenerate electron endpoint convergence failure in the old unsmoothed integration. The crossed endpoint map addresses the latter; neither failure should be removed from the evidence record.

This review independently reproduces the extension limitation in **unchanged** `allowed_opacity/allowed_capture.py`. At T=3 MeV, `(nu_n,nu_p,mu_e)=(20,5,5)` MeV, antineutrino E=1.8 MeV:

```
independent momentum integral: chi = 66.0406884968338 s^-1
frozen order24:                 chi = 66.0413018341791 s^-1
relative default error:              9.28726455e-6
frozen order96 relative error:       1.52029827e-7 .
```

The independent angular refinement 64→128 changes its answer by 4.4e-16 relative. Thus the reported default-order error is reproduced with a different outer variable and integration method. It exceeds the previous capture target of 3e-6 at this extension point. Existing passing selected capture states remain accepted; a general default24 combined-kernel scan near this endpoint is **not accepted** until a scoped repair is independently reviewed or the region is excluded/refined explicitly.

Nonrelativistic thermal tails can be numerically positive while sampling momenta outside a physically controlled NR expansion. Logarithmic positivity is not evidence of relativistic accuracy. No generic all-energy or all-degeneracy qualification is claimed.
