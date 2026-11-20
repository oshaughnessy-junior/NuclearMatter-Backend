# Independent modified-Urca contract review

**Accept the source audit, reaction bookkeeping and normalized degenerate energy reference in their declared scope. Absolute modified-Urca rates, widths, spectral coefficients and matching remain OPEN.** The author is not changed. A fresh independent probe gives **237/237 selected checks PASS**, 0.018 s numerical runtime. Two coarse outer-cutoff diagnostics fail the unchanged 3×10⁻⁸ fine-rule gate and remain explicitly excluded.

## Independent derivation and numerical test

Write f(x)=1/(1+eˣ). Its bilateral Laplace transform is π/sin(πs), 0<s<1: substitute z=eˣ and use the beta integral. The normalized exponentially tilted density at s=1/2 is

```
g(x)=e^(x/2) f(x)/π = 1/[2π cosh(x/2)]
Fourier[g](k)=sech(πk).
```

Thus multiplication of FIVE original one-fermion characteristic functions gives

```
J5(y)=π⁴ exp(−y/2) ∫₀∞ dk cos(ky) sech(πk)⁵.
```

This is independent of the author's J2/J3 convolution and closed polynomial evaluator. It uses neither the author's J2 nor J3. Fresh signed y=−21.3,−7.17,−0.083,1.234,6.789,19.7 and separate Fourier cutoffs8/10 agree with the closed J5 within 2×10⁻¹⁰. Bilateral transform tests at s=.19,.413,.79 independently recover [π/sin(πs)]⁵. Inversion gives the author's `[y⁴+10π²y²+9π⁴]/[24(1+exp(y))]`. In particular, J5(−y)=eʸJ5(y); signed arguments and five final-hole transformations are consistent.

For one emitted branch at zero affinity, integrate y²J5(y) for number and y³J5(y) for energy. Expand only for this analytic derivation and use ∫₀∞ yᵐ/(1+eʸ)dy=Γ(m+1)(1−2⁻ᵐ)ζ(m+1). With the even-zeta Bernoulli values ζ4/π4=1/90, ζ6/π6=1/945, ζ8/π8=1/9450, exact rational arithmetic gives

```
IE/π⁸ = [7!(127/128)/9450
          +10·5!(31/32)/945
          +9·3!(7/8)/90]/24
       =11513/120960.
```

A second outer quadrature over the **Fourier** kernel gives IN=191.5860297928814 and IE=903.1205157369873, agreeing with analytic values within 4.10×10⁻¹¹ relative; mean Eν/T=4.71391633677. Truncating at y28 instead of40 produces number error −1.019×10⁻⁷ and energy error −6.320×10⁻⁷. These two errors FAIL the same 3×10⁻⁸ gate; `coarse_failures.json` preserves them. No gate was loosened. Fine-cutoff results and the coarse exclusions are separate.

The T counting is conditional: five linearized fermion radial energies give T⁵; the energy delta gives T⁻¹; the neutrino number weight Eν²dEν gives T³. Hence T⁷ number and T⁸ energy only when angular factors/current/dispersion derivatives can be frozen on their Fermi surfaces. In a covariant description the neutrino spin sum supplies its linear energy to compensate the neutrino 1/(2Eν) measure. An arbitrary constant covariant amplitude would have different counting. This result does not specify the missing DOS/angular/current coefficient and supplies no arbitrary-degeneracy thermal rate.

## Inventory, symmetry, chemical signs and units

I reconstructed the author's saved 32 directed reactions with an independent stoichiometric charge table; each conserves baryon number, electric charge, Le and Lmu, and every inverse has exactly swapped initial/final multisets. Each labelled event has exactly one same-species baryon pair on one side, so the factorial measure is1/2 in both directions. Exchange interference remains in the coherent amplitude and does not replace this factorial. Per-spin FD products plus a complete spin sum require no additional spin2 factor. This agrees with [Shternin, Baldo & Haensel Eq2](https://arxiv.org/pdf/1807.06569), which explicitly supplies a spin sum and s=1/2. Their doubled cold emitted-branch expression does not justify doubling a microscopic inverse absorption coefficient.

For either spectator, µN cancels in each family. Anti emission has µn−µp−µl−µanti; neutrino emission has µp+µl−µn−µnu after µl+=−µl. These give µanti=µn−µp−µl and µnu=−µanti at trapped LTE. They do not imply transparent finite-temperature stationarity for the incomplete inventory. Material energy change is minus escaping energy with no additional Q.

The five-Fermi polygon condition must remain independent of the direct triangle. Fresh electron-only neutral momentum tests independently locate the proton-spectator polygon condition kFn≤4kFp, or Yp≥1/65 under that specific lepton-counting closure. This is a kinematic illustration, not a cold EOS prediction; nonzero muons change kFl. Tests below/above1/65 and fm⁻¹↔MeV conversion by ħc agree. A covariant six-external-particle number integral has E¹² from six measures, E⁻⁴ from delta4, and E⁻⁴ from the squared 2→4 amplitude, giving E⁴; adding emitted energy gives E⁵. Conversion is 1/(ħ(ħc)³) to fm⁻³s⁻¹ or MeV fm⁻³s⁻¹. The documented K/fm⁻¹/cgs cold source cannot be compared by treating kelvin as MeV.

## Primary-source and pinned-source audit

[Suleiman, Oertel & Mancini Eqs23,28,31 and appendices](https://arxiv.org/html/2308.09819v2) support the multi-diagram OPE treatment and phenomenological propagator regulator described in the note. I separately checked the [PDF Eq23](https://arxiv.org/pdf/2308.09819v2): both renderings show the repeated pl−pl numerator with a different denominator transfer. That ambiguity is real and remains unresolved; it is not a reason to code zero or silently assert a corrected operator. Their polarization measure must be reconciled with the amplitude normalization before absolute normalization is claimed.

[Alford, Haber & Zhang Eqs6–10 and SecIII](https://arxiv.org/html/2406.13717v1) define iW/2, a normalized Lorentzian on signed mass, and the mass convolution of a mean-field direct rate with fixed chemical potentials. My separate tangent-transform/Jacobian checks verify the half-width factor at new W=.031,2.7,19 MeV. This is a total charged-current prescription: adding a separate MU rate or dressed DU rate duplicates retained collision physics. The paper warns that omitting vertex corrections is unsuitable for the neutral-current Ward identity. Positivity of a Lorentzian is not a proof that the current positive-mass rate implementation covers signed spectral support.

[Sedrakian & Dieperink Eq69 and adjacent discussion](https://arxiv.org/pdf/astro-ph/0002228) give a≈.2 MeV⁻¹ in their pure-neutron microscopic setting, with frequency and temperature restrictions; their γ and plotted γ/2 must not be confused with an arbitrary regulator convention. This does not establish a universal n/p width for another EOS or composition. Thermal-tail direct opening, fixed-order spectator diagrams and a dressed-width resummation remain distinct calculations needing explicit matching.

The pinned `core_nu.py` bytes independently reproduce Git blob520d3b458c5eddbed1e1e1a5c6270dcc12643eb5. I inspected its murca body without importing JAX/production code: neutron coefficient contains kfe+kfm, proton coefficient kfe only, both use T9⁸; the only local nucleon-momentum guard is their nonzero product. No explicit proton polygon Heaviside is present there. The source labels and source limitations in the author audit are accurate. This is source inspection, not runtime parity or a production defect claim.

## Unchanged open gates

No strong NN amplitude, relative diagram signs/interference, normalized antisymmetrization implementation, off-pole shell integral, interaction-derived width, pairing calculation, or absolute modified-Urca rate is implemented here. The cold normalization comparator, new warm phase-space validation and consistent subtraction/resummation matching remain required. Neither the207 author checks nor these237 review checks reduce that physical uncertainty. No finite-T/cold GM1 forward claim is made.

`probe.py` imports ONLY the author's pure energy module under a unique module name; reaction inventory is read as JSON and source is read as bytes. No author driver executes. Before/after hashes of all author artifacts agree exactly. Numerical outputs, both failed coarse-cutoff diagnostics and frozen source hashes are retained locally. Paper renderings were verified online on2026-10-15; those renderings are not claimed to be byte-pinned paper downloads.
