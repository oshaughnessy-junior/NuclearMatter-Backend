# Read-only endpoint probe of the earlier capture checkpoint

15 October 2026. `allowed_opacity/allowed_capture.py` was left unchanged. This extension probe is outside its previously selected qualification sample. It tests the same nonrelativistic state used for the new crossed degenerate checks:

```
T=3 MeV, M=939 MeV,
O_n=Q=1.29333236 MeV, O_p=0,
mu_n=Q+20 MeV, mu_p=5 MeV, mu_e=5 MeV.
```

For antineutrino absorption the ordered baryons are p→n and `mu_lepton=−5 MeV`; for neutrino absorption they are n→p and `mu_lepton=+5 MeV`. The response and current model are unchanged. Frozen Gauss-Legendre energy/angle orders 12, 24, 48, 96 and 192 are compared to independently integrated `(q,E_e)` QUADPACK values. Results, absolute coefficients and exact state records are in `frozen_capture_endpoint_probe.json`; the reproducible driver is `probe_frozen_capture.py`.

| Absorption branch | E_nu (MeV) | Relative error at order 24 | Relative error at order 192 |
| --- | ---: | ---: | ---: |
| nu_e+n | 0.1 | −1.46e−11 | −6.66e−16 |
| nu_e+n | 1.0 | −1.60e−11 | +2.22e−16 |
| nu_e+n | 5.0 | −5.53e−14 | −2.22e−16 |
| anti-nu_e+p | 1.0 | −1.50e−11 | +2.88e−13 |
| anti-nu_e+p | 1.5 | +3.26e−7 | +5.13e−10 |
| anti-nu_e+p | 1.8 | **+9.29e−6** | +1.92e−8 |
| anti-nu_e+p | 2.0 | +1.15e−6 | +2.37e−9 |
| anti-nu_e+p | 3.0 | +4.04e−12 | +8.66e−15 |

The 1.8 MeV case exceeds the earlier capture check's 3e−6 integration target at default order 24. Its decreasing error under refinement is consistent with the same integrable electron-mass endpoint corner that appeared more strongly in the crossed channel. This **does not invalidate the earlier passing selected-state checks**; it identifies an extension-domain integration limitation that should be repaired or explicitly excluded before broader scans. The exact `E_e=m_e+x²` map repaired the analogous crossed-channel failure and is a concrete candidate for a separately reviewed capture update. No such update is made here.

The most suppressed subthreshold antineutrino tails can require nucleon momenta outside a physically controlled nonrelativistic regime. Numerical agreement of those formal-model tails is not an assertion of physical accuracy. This report evaluates the integrator, not an expanded validity domain for its underlying interaction model.
