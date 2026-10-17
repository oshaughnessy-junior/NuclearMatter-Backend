# Free relativistic thermal neutron decay and inverse three-body reference

**Actual separate differential and selected lepton-integrated spectral coefficients are implemented.** The qualified slice is a very dilute free npe gas at T=0.03 MeV, nB=1e-12 fm^-3, Ye=0.3, with selected Eanti=0.3 MeV spectral integration and four differential points. Full FD occupations, recoil, physical masses, the accepted full current and all 16 baryonic tensor components are used. This is not a complete antineutrino inventory: positron capture remains a separate kernel. Independent review is pending.

## Physical contract

Channels are n→p+e−+antiνe and p+e−+antiνe→n. Chemical potentials include rest mass. Bare physical masses are mn=939.56542052, mp=938.27208816, me=0.51099895069 MeV, and the weak-current mass is the distinct bare mean 938.91875434 MeV. No effective masses, mean fields or additional Q shift are used.

For prescribed Enu,Ee,c=cos(angle(e,anti)), let

\[
\Omega=E_\nu+E_e,\qquad q=|\boldsymbol k+\boldsymbol l|,
\qquad s=Q^2=m_e^2+2E_\nu(E_e-p_ec).
\]

The exact free invariant support is me²≤s≤(mn−mp)². The response is ordered **p→n** at positive Q=(Ω,q). The weak current Γ(+Q) is the reciprocal of the decay current Γ(−Q). The crossed lepton tensor is computed from the actual incoming electron and antineutrino current,

\[
L^{\mu\nu}_{\rm inv}=\mathrm{tr}[\slashed k\gamma^\mu(1-\gamma_5)
(\slashed l+m_e)\overline{\gamma^\nu(1-\gamma_5)}].
\]

This is not the positron-capture tensor. The underlying W has unaveraged target spin sums and invariant baryonic measures. No CVC component elimination or additional electron-spin factor is applied. The coefficients are

\[
\frac{d\chi}{dE_e\,dc}=\frac{G_F^2V_{ud}^2}{32\pi^2\hbar}
\frac{p_e}{E_\nu}f_e L_{\mu\nu}W^{\mu\nu}_{p\to n},
\]
\[
\frac{dj}{dE_e\,dc}=\frac{G_F^2V_{ud}^2}{32\pi^2\hbar}
\frac{p_e}{E_\nu}(1-f_e)L_{\mu\nu}\widetilde W^{\mu\nu}_{p\to n}.
\]

W uses fp(1−fn); tilde-W uses fn(1−fp), with reciprocal amplitudes. Both are evaluated separately, rather than deriving one coefficient by exponentiating detailed balance. Their independently evaluated logs satisfy

\[
\log(j/\chi)=-[E_\nu-(\mu_n-\mu_p-\mu_e)]/T.
\]

The collision term per active antineutrino occupation is C=j(1−fnu)−χfnu. j and χ are s^-1 after Ee/c integration; the differential units are s^-1 MeV^-1 per dcos. The one-helicity DOS is Eν²/(2π² ħc³), in fm^-3 MeV^-1. Empty-radiation production is DOS×j. Relaxation attenuation would be j+χ, not χ alone.

The channel, occupation inventory and crossing agree with [Guo et al., Sec. III.2, Eqs. (36)–(37)](https://arxiv.org/html/2006.12051v2). The full V/A, weak-magnetism, pseudoscalar and physical-q² form-factor primitives are shared with the accepted tensor/current references; this implementation is not independent of those source conventions. Coulomb, radiative, many-body, nuclear binding and medium effects are omitted. [Ando et al.](https://arxiv.org/html/nucl-th/0402100v2) explicitly distinguish the omitted Coulomb/radiative contributions from tree-level decay.

## EOS and numerical domains

`matched_npe` solves the genuine relativistic particle-only spin-two baryon density roots and the exact relativistic e± net-charge root. Muons are absent in this leaf. Forward n,p and neutrality residuals pass before kernel tests. For the main case, total chemical potentials in MeV are (mun,mup,mue)=(939.1130868115629,937.7943974977201,0.3683831307935174). Antibaryon diagnostics remain those of the frozen EOS.

The differential API retains exact invariant support for any positive Enu, including Doppler tails above the neutron-rest endpoint 0.7820077 MeV. Exact q=0 needs a separate coordinate limit and is explicitly rejected. Finite logs, ordinary representation flags and empty support are distinct; there are no positive floors.

The selected spectral rule resolves Ee around the exact neutron-rest shell for each lepton angle, using a width q sqrt(T/mn), multiple inner panels and explicit outer tail-width refinement. It rejects T outside [0.01,0.1] MeV, baryon degeneracy η>−8, and Enu above the rest endpoint. **Those guards are restrictions of the current integrated rule, not physical support cutoffs.** Even within them, only the listed cases have numerical qualification; they are not a general domain certificate. The Ee truncation is empirical and must be refined at new states.

## Checks and results

53 main checks plus 4 independent event-spectrum checks pass. The main experiment takes about 16 s; the independent event-spectrum experiment takes about 9 s. Each saves raw reports and checkpoints, with a 110/105 s work bound.

An independent differential reference boosts a Q-rest baryon sphere to the bath frame, uses explicit crossed external spinors and derives support and Jacobian from

\[
E_p^*=(m_n^2-m_p^2-s)/(2\sqrt s),\quad
E_p=\gamma(E_p^*+\beta p^*c^*),\quad
\frac{dE_p}{q}=\frac{p^*}{\sqrt s}\,dc^*.
\]

Its radial coordinate, support derivation and amplitudes differ from the q-axis energy trace. Shared gamma/current primitives are declared. At (Enu,Ee,c)=(.3,.993,0),(.3,.998,−.54),(.41,.883,.62),(.81,.523,.9), separate j and χ agree within 3.93e-8 relative; baryon shells and anti-KMS pass. The final point has **Enu=.81 above the rest endpoint**, with finite j=3.73099285e-20 and χ=3.47283685e-22 s^-1 MeV^-1 per dcos. It is not spuriously removed.

At Enu=.3, the refined main-bath spectrum is

- j=1.6622811423061443e-6 s^-1;
- χ=6.405576907923156e-16 s^-1;
- independent Q-rest/spinor spectral integration agrees within 1.04e-9 relative;
- outer orders 10/12 versus 14/16 differ by 2.94e-11;
- Ee width 9→12 and internal baryon order16→24/tail50→70 changes are below 5e-13.

The independent spectral reference uses the same outer Ee/c panels but independently performs the baryon phase integral and amplitude. Outer refinement is therefore a distinct necessary check, not claimed independence of every numerical choice.

Multiplying j by the one-helicity DOS and dividing by neutron density gives thermal dΓ/dEnu. Compared with independently integrated exact-rest dΓ/dEnu=.0014092341001856038 s^-1 MeV^-1, relative differences are −2.3273e-4 (T=.1), −5.3758e-5 (.03), −1.8124e-5 (.01), at fixed nB=1e-12,Ye=.3. These are finite-temperature/occupation differences, not numerical error estimates.

An independent momentum integral gives <mn/En>=.9999521074261754 at T=.03. For unblocked vacuum decay the **future total-energy-integral target** is <mn/En>Γrest=.0010533341716 s^-1 using the accepted tree-level lifetime949.32086547s. **The current selected-energy results do not test the total time-dilated lifetime.** Full Enu number/energy moments remain open.

A deliberately prescribed extreme negative electron chemical potential yields finite logχ≈−100142.765 with ordinary χ=None and `reason=underflow`. This tests representation only, not a matched neutral bath. Invalid finite/positive controls are rejected.

## Preserved failures and open gates

`initial_pilot_failure.json` preserves an upstream full-matter request at T=.03,Ymu=0 that failed because its unnecessary muon pair susceptibility underflows. The explicit npe factory avoids that unsupported muon state; no frozen EOS was changed.

`initial_Qrest_coarse_failure.json` and its exact validation source preserve **eight failed differential comparisons** from uniform order80 Q-rest cosine quadrature. Thermal baryons occupy a very thin polar cap; that uniform reference grid misses it by orders of magnitude. The refined independent reference resolves the affine Ep(cstar) cap with thermal-energy panel boundaries. No physical kernel coefficient or acceptance tolerance was changed to repair this quadrature failure.

Independent review is required before promotion. Total neutrino-energy integration/time-dilated lifetime, broader thermal/density/degeneracy domains, near-endpoint integrated tails, exact-q=0 limit, full relativistic antineutrino capture+decay inventory, and any GM1/medium crossed kernel remain open. There is no measured-lifetime calibration or physical-uncertainty estimate from numerical agreement.

Reproduce from repository root, in order:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/relativistic_crossed_thermal/validate.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/relativistic_crossed_thermal/event_spectrum.py
```
