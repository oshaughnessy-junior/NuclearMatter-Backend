# Independent review: optional normal cold modified-Urca recipe

**ACCEPTED for the declared source-labelled normal, degenerate, equilibrium energy-emissivity recipe and its cold constant-matrix angular geometry.** This is not an independently derived absolute modified-Urca matrix element, a general finite-T kernel, a new EOS/beta-equilibrium solution, or a physical error certificate. Frozen source SHA256 `c9a54893be63a0a0cf5701626e23080524bfa30826eea81e5070ec60648c8ef6` and manifest `ca384a4f6d2952ad9536520995abcca1a3bf2ae591b9bcb8cfee249cdae76a27` match. All author files remain unchanged.

Command: `/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_cold_murca_normal_reference/probe.py`. Final independent report: **107 PASS, zero failures**, .0032s. Counts include provenance and stoichiometric checks; they are not107 independent absolute-rate benchmarks. An initial reviewer-fixture failure remains preserved verbatim, as described below.

## Primary-source audit

[YKGH2001 Eq140, Eq142 and Sec3.4d](https://arxiv.org/pdf/astro-ph/0012122) support the chosen8.1e21 cgs coefficient, proton-density factor, mass powers, alpha1.13/beta.68, and lepton Fermi-velocity correction. The proton/neutron rescaling assumes equal angle-averaged branch matrix elements. Thus numerical agreement with8.55e21 or a different density-dependent alpha is not required or inferred. Published coefficients and interaction approximations do not warrant the precision of printed numerical outputs.

[Gusakov2002 Eqs10–15](https://arxiv.org/pdf/astro-ph/0204334) define the five-material-vector angular integral, its neutrino solid angle, neutron region and proton polygon support. The PDF text extraction is ambiguous in the lower-region superscript; my acceptance of the coefficient8 follows the independent absolute geometry below, not an OCR reading. Both branches neglect neutrino momentum in the material momentum delta function. Finite-neutrino-momentum support changes and thermal opening are outside this cold geometry.

## New independent absolute angular reduction

The author integrates a positive overlap of two- and three-vector radial densities. My calculation instead uses the **longitudinal projection** of five independent isotropic vectors. A vector of length li has zi uniform on [−li,li]. Their sum has a five-box-spline longitudinal density

fZ(z) = sum_S (−1)^|S| [z+sum_i li−2sum_(i in S) li]_+^4 / [4! product_i(2li)].

Isotropy gives fZ(z)=2pi integral_|z|^infinity r rho(r)dr, hence rho(0)=−fZ''(0)/(2pi). Multiplying by all five material solid angles and the independently integrated neutrino4pi yields

A = −32pi^5/product_i(li) * sum_S (−1)^|S| [sum_i li−2sum_(i in S)li]_+^2.

This finite32-subset expression uses70-digit Decimal arithmetic before the final pi factor. It shares neither overlap panels nor adaptive quadrature with the author. Relative to the author's128pi^5/(product k) overlap integral, it independently establishes both the absolute Fourier/solid-angle normalization and the phase-space remapping, rather than calibrating either.

Eight NEW length triples test low/middle/forbidden proton regions and neutron geometry, with three overlap groupings and two common length rescalings. The largest scaled angular discrepancy is6.29e−14 at the unchanged3e−12 gate. Dimension is k^-3, i.e.fm^3 for k in fm^-1. Three new sub-edge gaps down to3.1e−5fm^-1 obey the independently normalized quadratic outer threshold at a3e−10 gate. The exact outer boundary is zero. G=(Ap/An)(kp/kn)^2 therefore has the declared piecewise values, including (3−kn/kp)/2 in the lower region. Replacing8 by8/3 would fail the independent absolute reduction and continuity at the internal boundary. This validates geometry with a constant matrix approximation, not an angularly varying strong amplitude.

The initial reviewer script demanded exact zero using `edge=3*.913+.247`; this evaluates to a point infinitesimally below the exact decimal sum. Its70-digit geometry is consequently1.396e−27, not exactly zero. `initial_probe.py.txt` and `initial_results.json` preserve that80PASS/1FAIL result. The corrected test uses an exact Decimal boundary, retaining a zero tolerance, and separately checks the author's own computed floating boundary returns zero. This is a reviewer fixture correction, with no author code or physical tolerance changed. Resolving relative rates within machine precision of a geometric boundary is not claimed.

## Fresh recipe and material accounting

I solved two new lepton-neutral fixtures by rooting total lepton chemical potential, rather than the author's electron-kF root: (nB,Yp,T)=(.271,.0591,4.31e8K) and(.182,.0417,2.37e8K), with prescribed Landau/bare ratios.82/.77. The resulting chemical potentials are141.267517338570 and116.967867494747MeV. Independent B-spline angular factors and direct source-labelled emissivity algebra reproduce all four channels; totals are2.98393256912720e18 and1.53357783793691e16erg cm^-3 s^-1. These are newly neutral lepton fixtures with supplied nuclear affinity, not solved nuclear EOS states.

Landau masses enter the density-of-states prescription, and vF_l=HC kl/sqrt[(HC kl)^2+ml^2] enters each flavor. The proton branch includes the ratio of ACTUAL Landau masses, including bare neutron/proton masses; using only their supplied dimensionless ratios would change the chosen convention. kF units, T9 conversion and cgs output match the source-labelled formula. Bare/Dirac/Landau masses are not interchangeable.

All eight spectator/flavor/direction entries conserve baryon number, electric charge and total flavor lepton number. The fields called `delta_material_Ye_per_event` and `delta_material_Ymu_per_event` are integer **stoichiometric number increments**. An actual abundance-fraction source must multiply by event rate per baryon; a single event in a finite parcel changes Y by1/NB, not by1. This naming advisory does not affect the present interface's explicitly assumed zero net composition drift. Its number-rate cancellation is an equilibrium assumption, not an independently integrated number moment. Positive summed emissivity removes material energy directly; adding a separate reaction-Q heating term would double count within a rest-inclusive material energy equation.

## Retained limitations

Charge closure, matching lepton chemical potentials and a supplied nuclear affinity are checked. The code does not calculate that affinity from an EOS. Its chosen kn>=1.68, vF_N<=.8 and kBT<=.03 minimum kinetic scale admissions constrain research use; none is a demonstrated physical accuracy bound. Strong interaction normalization/correlations, paired suppression, nonzero chemical imbalance, absorption spectra and numbers, finite-T/full-momentum phase space, and DU/MU matching remain open. The absolute emissivity coefficient is adopted from the declared prescription rather than obtained from the campaign's separate strong/weak amplitude chain. Existing NMB cold prescriptions and their previous qualification/failure records remain untouched.
