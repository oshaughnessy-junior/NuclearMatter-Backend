# Independent physics review: selected cold capture quadrature

15 October 2026. **Accept the selected numerical repair within the existing free common-mass NR allowed-current model.** No blocker at the three author states or the additional spectral point below. This does not qualify a general cold rectangle, interacting GM1, full Urca emissivity or a new physical broadening mechanism.

Reviewed frozen `cold_capture/resolved.py` SHA256 `d988860e717ce3a08d8fc2d59a3bdc91c506be80ffb44250e6c55cd67b5354a1`, source note, study/refinement reports and upstream response/current. Author files were not changed. `cold_capture_probe.py` and raw JSON are separate review-owned artifacts. **23 independent checks pass in1.98s**, with numerical gates unchanged. These check specific equations and selected forward coefficients, rather than counting the author's87 checks again.

## Kinematics and occupied Fermi volumes

Choose q along z. With common kinetic mass M and `w=omega−(offset_p−offset_n)`, the energy delta fixes `pz=(2M w−q²)/(2q)`. Initial kinetic energy is at least `emin=pz²/(2M)`. Neutron occupation gives `t<=nu_n`; proton vacancy gives `t+w>=nu_p`. Therefore the exact cold interval is

`I=max(0,nu_n−max(emin,nu_p−w))`.

This establishes the correct direction of initial/final Pauli edges. Independently, the initial transverse disk has squared radius `max(0,pFn²−pz²)` and the blocked final disk `max(0,pFp²−(pz+q)²)`. Their allowed difference gives `S=M/(2pi q(hbar c)³)*(allowed transverse radius²)`, equivalent to `M² I/(pi q(hbar c)³)`. Nine geometry cases include closed and open support for Enu=.03,.32,.9MeV at varied outgoing electron energy/cosine. Three nonzero cases approach their cold values at T=1e−6MeV. This checks finite incident energy directly; the zero-neutrino cold triangle is insufficient.

The author's conditions `emin=nu_n` and `emin+w=nu_p` correctly mark, respectively, a support edge and the change of which lower bound controls the interval. **They are not an exhaustive enumeration of cold support boundaries:** the second support inequality also reaches zero at `nu_n+w=nu_p`, i.e. `Ee=Enu+mu_n−mu_p`. Electron vacancy independently turns on at Ee=mu_e at T0. The existing energy/Fermi panels and demonstrated convergence resolve the selected cases; no domain is truncated by those root markers. Thus this is a scope clarification, not a rejection of the reported coefficients. A general cold-domain integrator should panel the energy-budget edge explicitly and establish root completeness; the current finite scan/sign-change finder can miss tangencies or multiple roots within one scan interval.

## Finite-T response and coefficients

Six held-out response tests independently integrate `f_n(t)[1−f_p(t+w)] dt` in initial kinetic energy, from the independently reconstructed longitudinal minimum, and repeat for the reversed pair/negative transfer. Log scaling and separate Fermi breaks make the occupation/vacancy signs visible. Agreement with the analytic response is within about1.6e−13 relative, including activated configurations around the electron Fermi surface. This verifies the selected physical FD opening without adding a width, smoothing the support or shifting the fixed offsets.

For capture, `q²=Enu²+pe²−2Enu pe c`, `omega=Enu−Ee`. The change of variable gives `pe dc=−q dq/Enu`; reversing limits produces the positive `Ee q/Enu` measure in the adaptive method. Direct cosine integration retains `Ee pe`. Both methods use the same physical lower endpoint Ee=me and the same frozen allowed angular factor and K. Combining K's dimensions with `dEe Ee pe S` gives s^-1. There is no new absolute normalization calibration.

The two coefficients separately integrate electron occupation/reverse response for j and vacancy/forward response for chi. KMS is checked afterwards. This avoids using detailed balance to manufacture agreement. The adaptive method samples a logarithmic scale; the other method sums weighted logs directly. No physical floor is imposed. Sampling a scale is not a guarantee of robust arbitrary-state integration, and finite upper-tail agreement is not an analytic infinite-range remainder bound. Both are acceptable selected-point numerical qualifications here.

## Independent spectral point and original failure

At the same calibrated nB=.01,Yp=.10 Hamiltonian, **T=.01MeV, Enu=.045MeV** is outside the author's E=3T point list. Adaptive y/q order72, tail160 and direct cosine64/energy32, tail180 give:

| coefficient | independent-coordinate value [s^-1] | fractional coordinate difference |
| --- | ---: | ---: |
| j | 1.62326767385904e−13 | 5.33e−14 |
| chi | 1.46667342670747e−11 | 5.68e−14 |

Both pass the original3e−6 gate. The independently calculated j and chi obey KMS with the EOS affinity, within the original3e−10 logarithmic gate.

At **the original T=.01, Enu=.03 failure**, a fresh replay of the unchanged mapped72 coefficient remains **−3.60225856e−4 relative** to the new resolved coefficient, for j and chi separately. It still fails3e−6; the old implementation has not been retrospectively promoted. The new resolved values are j=1.42714116629650e−13 and chi=2.87718968203616e−12s^-1. The original failure artifact retains SHA256 `7cf53b0e10dc201fd2dc59ae7acc9e5ca44b9d0525d38ada4ad9399627fc4b96`.

## Accepted scope and next gates

Accept the author T=.03 core comparison, exact previously failed T=.01 state and colder T=.007 selected results as evidence of **a numerical resolution repair in an unchanged response**, together with the additional Enu=.045 point above. The local offsets were tuned once at T0 and held fixed; no claim about real neutron-star beta equilibrium follows. Forward coordinate agreement is numerical evidence inside a shared physical ansatz, not external model validation. General cold support coverage, analytic tail control, relativistic/current corrections, correlated matter and energy-integrated total Urca remain separate gates.
