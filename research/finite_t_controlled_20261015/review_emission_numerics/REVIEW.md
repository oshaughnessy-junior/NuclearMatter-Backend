# Independent numerical review: log-domain emission-only inventory

2026-10-15. **Accept the selected emission-only normalization and representation claims.** All **40/40 independent checks pass** in21.65 s, below the90 s experiment bound. Acceptance is limited to the declared T=0.3 MeV, commonM=10⁷ MeV, nn=7×10⁻⁵ and np=3×10⁻⁵ fm⁻³, prescribed total μe=0 bath and the named positron-capture plus neutron-decay antineutrino inventory. This is not a neutral EOS state, generic opacity or trajectory qualification.

Author `emission_only/log_emission.py` SHA256: `f73fd3504498b696614fb5153362f5d27816e659a8cb5bb43d4467ccbde28207`. Source and preserved failures have identical hashes before and after review. No author/upstream files were edited.

## Forward state and independent static normalization

Independent dimensionless kinetic-energy integrals reconstruct both actual FD number densities with a worst relative residual5.56×10⁻¹⁶. The physical neutron–proton Q offset is retained, while μe stays prescribed at zero rather than being replaced by a charge-neutral electron root. The huge common mass is a mathematical dilute/static normalization experiment. It is not the physical nucleon mass, an interacting EOS, or a measured neutron-lifetime calibration.

The independent static reference integrates **electron momentum** with adaptive scalar quadrature. It does not call the author `target_moments`, static `spectral`, endpoint variable map or its numerical energy panels. With Ee=sqrt(p²+me²), Ee pe dEe=p²dp, so the allowed-current per-neutron moment is proportional to ∫dp p² Eν^(2+k) times the positron occupation (capture) or electron vacancy (decay). Eν=Ee+Q for capture and Q−Ee for decay. The prefactor is reconstructed from the declared GF,Vud,gA and ℏ benchmarks; target nn multiplies the result. The finite-T electron blocking is retained in decay.

Independent static values are:

| Contribution | Number [fm⁻³ s⁻¹] | Energy [MeV fm⁻³ s⁻¹] |
|---|---:|---:|
| Positron capture | 2.52214394452×10⁻⁷ | 6.74086798680×10⁻⁷ |
| Neutron decay | 6.87430504465×10⁻⁸ | 3.25831755790×10⁻⁸ |
| Complete named sum | 3.20957444899×10⁻⁷ | 7.06669974259×10⁻⁷ |

Each finiteM author number/energy moment agrees independently with this static calculation within6.27×10⁻⁷, passing the unchanged3×10⁻⁵ gate. Separate channel comparisons prevent the combined sum from hiding an omitted contribution. New allowed-region spectral points Eν=0.23 MeV for decay and2.6 MeV for capture agree with the static spectral coefficient within6.51×10⁻⁷. Both channel logarithms are evaluated at both points; the combined coefficient is checked against their explicit logsumexp.

## Different finite-M outer quadrature

A second complete physical integration uses **log(Eν)** rather than the author square-root endpoint coordinate. It independently generates geometric threshold panels on both sides of Q−me and Q+me, with spacing proportional to sqrt(T/M), includes a linear panel down to Eν=0, and extends to32 MeV. Every positive energy node evaluates both physical channel kernels. The finite-mass tails and the nominal static gap remain in the quadrature; no static support hole is imposed.

Positive quadrature measures, channel coefficients and final moments remain logarithmic. The log-coordinate Jacobian adds one power of E to the measure; one-helicity DOS is included exactly once. Outer orders10 and16 agree within7.80×10⁻⁸. The independent16-node moments agree with the author20-node mapped moments within6.09×10⁻¹² for every separate channel and the sum. The independent refined moments also agree with the momentum-static normalization within6.27×10⁻⁷. This validates a selected full-energy integration using a different outer coordinate/panel strategy while sharing the same frozen finite-M response/current kernels.

The source was inspected for absence of an ordinary χ positivity requirement: it directly requests logarithmic capture and crossed-decay coefficients, selects their j logs, and never converts χ to an ordinary float. Ordinary conversion occurs only in the explicit representation function after log-domain integration. This is an emission-only interface, not a repair of the old transfer wrapper.

## Extreme-tail representation and preserved exclusions

At Eν=0.8313293913178066 MeV, the same prescribed bath yields finite log j values approximately−12489427.3634 for capture and−55933.9510 for decay/combined. For every channel and the sum, ordinary value is `None`, `representable=False`, and reason is `underflow`. No coefficient floor or channel omission is used. These checks establish preservation and explicit representation of the log result. They do **not** independently qualify extreme-tail relative accuracy of the upstream asymptotic response.

The author initial import failure, numpy-bool JSON serialization failure and completed pre-cast partial output remain unchanged. The original `quantum_energy_moments/static_inventory_failure.json` also remains unchanged: it used the **different neutral T=2 MeV bath** and strict positive j/χ transfer wrapper. That original gate is still excluded. The new T=0.3,prescribedμe=0 emission-only comparison does not retroactively turn the old wrapper into a domain-safe opacity or transfer interface.

No new failed check occurred in this review. General prescribed-μ baths, other flavors/inventories, nonempty radiation, EOS coupling, current corrections, correlated/interacting response and external physical validation remain separate tasks. The physical review may add interpretation conditions beyond this numerical acceptance.

Reproduce with `python outputs/nmb-finite-t-20261015/review_emission_numerics/probe_emission.py`. Raw checks, alternate energy edges, logarithmic channel moments, independent static values, witness flags and before/after hashes are in `results.json`.
