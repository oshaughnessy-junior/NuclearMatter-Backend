# A controlled finite-T direct-Urca phase-space threshold study

15 October 2026. This leaf supplies a **physical occupation/kinematic threshold demonstration inside a named quasiparticle ansatz**, with nine qualified spectral points, independently checked quadrature and a preserved colder integration failure. No sigmoid, prescribed smoothing width or collisional broadening is inserted. No GM1/interacting-EOS, full emissivity or production-opacity claim is made.

## Why this model and what is held fixed

Use spin-two free NR neutron/proton dispersions with common kinetic mass M=939 MeV,

```
epsilon_n(p)=O_n+p²/(2M),  epsilon_p(p)=O_p+p²/(2M),
epsilon_e(p)=sqrt(p²+me²),  epsilon_nu=pnu .
```

Charged leptons use the existing relativistic pair-equilibrium density inversion. Set nB=0.01 fm^-3, independent Yp, and charge neutrality ne,net=np=Yp*nB. The composition contains no muons, clusters, pairing or nuclear interactions. The cold electron chemical potentials are 61–65 MeV, below the muon rest energy; nevertheless electron-only content remains an explicit model choice.

A vacuum-Q free gas would not be in cold beta equilibrium near the desired proton-fraction triangle. Instead, for each selected composition, define **one local externally prescribed constant offset**:

```
O_p=0,
O_n=Delta=EFp + mu_e(T=0) - EFn .
```

The kinetic Fermi energies are pFi²/(2M). This makes mu_n=mu_p+mu_e at exactly T=0 for that model. Once chosen, both offsets are held unchanged as temperature varies. It is a legitimate local ideal-quasiparticle Hamiltonian with constant species energies; it is not an interaction potential inferred from an EOS. Different compositions here have different prescribed offsets, so the plotted composition sequence is a family of local models, not one density-functional EOS sequence. No composition derivative across that family is taken.

| Yp | pFn−pFp−pFe [MeV] | fixed Delta [MeV] |
|---:|---:|---:|
|0.10|+4.88882998|54.44701850|
|1/9|0|56.84420535|
|0.12|−3.70899115|58.65611526|

The neutron pF/M is about 0.135, proton pF/M about 0.065–0.069, and T≤0.3 MeV. These are NR expansion diagnostics, not quantitative error bounds for the omitted current/dispersion corrections. Uniform nuclear matter at this density is not asserted to be the real preferred phase.

## Cold triangle and finite-neutrino distinction

At beta equilibrium, T→0 and negligible neutrino momentum, all charged participants approach their Fermi surfaces. Momentum conservation requires

```
pFn <= pFp+pFe .
```

With electron-only neutrality and equal spin degeneracy, pFe=pFp, so `(1-Yp)^(1/3)<=2Yp^(1/3)` gives Yp≥1/9. This is a limiting geometric statement. A finite-energy incident neutrino can permit absorption below that triangle by supplying energy and momentum, even at T=0.

For the selected Yp=0.10 cold model, the minimum-energy absorption onset occurs with the neutron and electron on their Fermi surfaces, the extra energy assigned to the slower proton, and favorable collinear momenta. In the selected low-velocity regime this gives

```
pFn = pFe + Ecrit + sqrt(pFp²+2M Ecrit),
Ecrit = B/[A+sqrt(A²-B)],
A=M+pFn-pFe,  B=(pFn-pFe)²-pFp² .
```

The stable small-root expression gives **Ecrit=0.30891898 MeV**. `cold_support.py` independently maximizes the exact cold scalar-response interval over the allowed electron-energy range and momentum-transfer interval. Five checks confirm zero support at Eν=0.03,0.09,0.30 MeV and nonzero support at 0.32,0.9 MeV. This is a finite-energy onset for this local model, distinct from the zero-neutrino 1/9 limit.

## Actual finite-T calculation

The evaluated channel is electron capture and its inverse absorption:

```
e−+p -> n+nu_e,    nu_e+n -> p+e− .
```

Return spontaneous j and true chi in s^-1 for `C[fnu]=j(1-fnu)-chi*fnu`. The leading allowed V0/axial-spin current, electron mass, exact common-mass NR recoil and all final-state FD blocking are retained by the existing qualified response/optical leaves. This leaf does not sum neutron decay/antineutrino channels or integrate neutrino number/energy moments.

For absorption, omega=Eν−Ee transfers energy into n→p baryons, q=|pν−pe| and the response uses the **same** fixed offsets as the thermodynamic state. The thermal broadening arises from occupied/vacant states in the exact phase-space integral. There is no finite quasiparticle lifetime, modified-Urca diagram or collision width in this calculation.

Use Eν=3T and T=0.03,0.1,0.3 MeV. Warm chemical potentials are solved at fixed species densities and offsets. The recorded affinity `mu_e+mu_p-mu_n` is nonzero: approximately −0.00029 to −0.0351 MeV over the selected points. It is reported, not retuned away. The coefficient ratio obeys

```
log(j/chi)=-(Eν-affinity)/T .
```

This channel/inverse detailed balance is not a statement that the transparent matter has reached kinetic beta equilibrium. That requires comparing the independent electron-capture and neutron-decay emission channels.

## Qualified numerical evidence

Run the core study and cold support check:

```
python3 outputs/nmb-finite-t-20261015/urca_threshold/threshold_study.py
python3 outputs/nmb-finite-t-20261015/urca_threshold/cold_support.py
```

**48/48 core gates pass** for nine states. Each state has independently reconstructed scalar-momentum neutron, proton and net-electron densities; capture order/tail refinement; channel KMS; and a separately integrated optical coefficient. The latter uses y=(Ee−mu_e)/T, q instead of lepton angle, adaptive scalar quadrature and a factored common logarithmic scale. It does not reuse the optical panel generator or energy nodes. The independent comparison shares the physical response/current equations and does not validate their omitted corrections.

The core largest independent density error is about 2.54e-12; the largest capture order/tail change is 4.62e-10. Independently integrated chi agrees within about 1e-12. Exact measured values, source/dependency hashes and diagnostics are in `qualified_results.json`; these are pointwise results, not a filled domain certificate.

| Yp | chi/T² at T=.03 | at T=.1 | at T=.3 |
|---:|---:|---:|---:|
|0.10|567.124|1.48407e6|5.54018e6|
|1/9|7.56811e6|7.60848e6|7.72533e6|
|0.12|9.55627e6|9.45459e6|8.87719e6|

Units are s^-1 MeV^-2. Dividing out T² is a useful degenerate spectral scaling diagnostic; it is not a fitted suppression law.

At Yp=.10,T=.03,Eν=.09 MeV, the cold absorption support is exactly absent in this model, yet finite-T chi is positive, about 0.5104 s^-1. This cleanly demonstrates thermal occupation enabling a cold-forbidden channel. At T=.3,Eν=.9, the neutrino itself is already above the cold finite-energy onset; the much larger rate combines thermal and finite-neutrino kinematic effects. The plot explicitly shows this distinction.

`threshold_selected.png/pdf` includes only core-qualified points; connecting lines guide the eye across the named local models. They are not empirical transition fits.

## Colder failed extension is retained

The optional `--cold-extension` adds T=.01 MeV and intentionally returns failure if its gates remain unmet. The original failed run is preserved as `initial_cold_extension_failure.json`, with its then-current script in `initial_study_with_cold_extension.py`.

At Yp=.10,T=.01,Eν=.03, density and algebraic KMS still pass, but capture48→72 changes by **0.6623%**, and order72 differs from the independent scaled adaptive integral by **0.0360%**, exceeding the unchanged 3e-6 gate. This point is excluded from the demonstration. A finite j/chi and correct KMS do not establish integration accuracy. No rate is set to zero or repaired by inserting an artificial physical width.

The next numerical task is a separately reviewed cold/degenerate optical panel scheme resolving the actual FD-product saddle and kinematic support edges, with log-scaled integration. Higher order alone is not a domain proof. Extreme log-response tails and ordinary-rate underflow also remain explicit boundaries.

## Sources inspected and next physics steps

- [Reddy, Prakash & Lattimer 1998, Eqs.16–30](https://arxiv.org/html/astro-ph/9710115v1) supplies the nonrelativistic phase-space response and its momentum threshold. It explicitly distinguishes the free-gas composition problem and finite-energy/temperature threshold behavior. Our constant-offset prescription is our declared controlled ansatz, not their interacting calculation.
- [Alford et al., beta equilibrium under merger conditions, Secs.II,IV,VI.3](https://arxiv.org/html/2108.03324v3) motivates full phase-space integration below the cold threshold and distinguishes transparent kinetic equilibrium from inverse-channel detailed balance. Its relativistic RMF rate results are not reproduced by this NR toy.
- [Lattimer et al. 1991, Direct URCA process in neutron stars](https://doi.org/10.1103/PhysRevLett.66.2701) is the foundational triangle-threshold reference, linked and discussed by the two inspected primary papers. Direct publisher/NASA fulltext access failed in this session; no unseen equation from it is used.

Next: independent code/physics review of this local-offset construction and the cold finite-energy onset; resolve the preserved cold quadrature failure; add the crossed antineutrino channel with the same offsets; validate number/energy moments before any thermal evolution. For dense interacting applications, replace the local model family with one thermodynamically consistent EOS and its compatible dispersion/current response. Do not convert these spectra into a universal sigmoid, infer collision broadening, or transplant them into GM1 by changing a Q value.
