# Elementary static strong OPE Born reference

This implements a **strong two-nucleon building block**, not a modified-Urca matrix element or rate. It supplies an explicit spin/isospin operator, coherent fermion exchange, analytic checks and convention-matched LO elastic Born cross sections. No weak vertex, intermediate nucleon propagator, finite-temperature collision integral, width, short-range fit, pairing or production change is included. Independent review is pending. Primary-source audit and exact links are in SOURCES.md.

## Defined operator, coupling and states

Natural units `hbar=c=1`; momenta and masses are MeV. A one-nucleon basis is `|p,s,t>`, `t=(n,p)` with `tau3=diag(+1,-1)`, and `<p',s',t'|p,s,t>=(2π)³δ³(p'-p)δss'δtt'`. The literal Pauli matrices have eigenvalues ±1, not ±1/2. For example `tau_minus=(tau1-i tau2)/2` maps n to p. The strong interaction conserves total charge and allows a neutron and proton to exchange their isospin labels.

Define the stripped two-body matrix element by `<34|V|12>=(2π)³δ³(Pf-Pi) V(q)` for the direct labelled graph, with q=p3-p1 and the other line transfer −q. The complete, unregulated static momentum operator is

```
V(q) = -C [(sigma1·q)(sigma2·q)] (tau1·tau2)/(q²+m_pi²)
C = f_piNN²/m_pi².
```

`V(q)` has units MeV^-2, consistent with its Fourier transform to an energy potential. This is the standard static derivative-pion result; Friman–Maxwell SecIII Eq10 and Savage Eqs13–15 support its transfer structure and sign. **The literal zero numerator in SOM23 Eq23 is not used.** [Original FM79](https://adsabs.harvard.edu/pdf/1979ApJ...232..541F), [explicit Savage derivation](https://archive.int.washington.edu/users/mjs5/Class_560/lec560_3/node2.html).

The selected reference is `f_piNN=1.01,m_pi=140MeV`, as specified by SOM23, with a common nucleon mass M=939MeV only for the LO elastic normalization. The common pion mass is an isospin-symmetric model; no charged/neutral pion splitting is included. The relation to a pseudoscalar coupling is **our declared on-shell equal-M NR conversion** `g_piNN=2M f_piNN/m_pi=13.5484285714`. It is not the pion decay constant. In a leading chiral convention `f_piNN/m_pi=gA/(2F92)=gA/(sqrt(2)F132)`; changing conventions must change the coefficient consistently. The numerical choice is not calibrated to NN data or asserted equal to another source's parameter set. [SOM23 Eqs22–24](https://arxiv.org/html/2308.09819v2), [independent GT convention](https://arxiv.org/pdf/nucl-th/0504067).

The full `q_i q_j` momentum operator is retained, including its coordinate-space contact distribution. No arbitrary short-range subtraction, form factor, cutoff or contact counterterm is silently attached. It is a simple Born interaction model, not a renormalized/iterated NN amplitude or the full Friman–Maxwell short-range prescription.

## Coherent exchange and charge channels

Internally the 16-dimensional matrix order is `(s1,s2) ⊗ (t1,t2)`, each four-state pair ordered `00,01,10,11`. Define spin swap Ps and isospin swap Pt. For k=p4-p1,

```
A(q,k) = V(q) - (Ps ⊗ Pt) V(k).
```

There is no sqrt2 factor in this **labelled external amplitude**. Equivalently, matrix elements between normalized antisymmetric incoming/outgoing states with distinct momenta give direct minus exchange; each normalization and each Wick contraction must be counted once. `P A(k,q)=-A(q,k)` and `A(-k,-q) P=-A(q,k)` verify outgoing/incoming fermion exchange. The latter uses total momentum conservation to map transfers; arbitrary q,k alone are an algebra test, not a scattering energy shell.

The isospin operator in `(nn,np,pn,pp)` is

```
[1  0  0 0]
[0 -1  2 0]
[0  2 -1 0]
[0  0  0 1].
```

Its total-isospin eigenvalues are +1 for I=1 and −3 for I=0. For fixed initial/final charge labels the four-spin matrix is

```
A_channel = -C [d S(q)/Dq - e Ps S(k)/Dk],
S(q)=(sigma·q)⊗(sigma·q), Dq=q²+m_pi².
```

| Channel | d: direct | e: exchanged final labels |
|---|---:|---:|
| nn→nn, pp→pp | 1 | 1 |
| np→np, pn→pn | −1 | 2 |
| np→pn, pn→np | 2 | −1 |
| nn→pp | 0 | 0 |

In np scattering the exchange graph contains charge exchange; treating the nucleons as labelled distinguishable particles and dropping that graph would omit an allowed amplitude. Conversely, np→np and np→pn over complete outgoing angles describe alternative labelings of the same set of physical final states and must not be added as independent total reactions. Exchange interference remains part of the coherent square.

## Independent spin traces

For unpolarized fixed charge species, average **only four initial spin states**, sum four final spin states, and do not average isospin:

```
Dspin(q) = Tr[S(q)†S(q)]/4 = q^4
Xspin(q,k) = Tr[S(q)† Ps S(k)]/4
           = (q·k)² - q²k²/2
<|A_channel|²>spin
 = C²[d²q^4/Dq² + e²k^4/Dk²
      - d e {2(q·k)²-q²k²}/(Dq Dk)].
```

These follow from `Tr[(sigma·a)(sigma·b)]=2a·b` and the four-Pauli trace. The code separately constructs Kronecker-product matrices, literal indexed spin/isospin contractions, and these scalar traces at seven seeded heldout pairs of generic three-vectors. A further full initial spin-and-isospin average is

```
Tr[A†A]/16 = C² [3(q^4/Dq²+k^4/Dk²)
                +3/2 {2(q·k)²-q²k²}/(Dq Dk)],
```

using `Tr(tau_dot²)=12` and `Tr(tau_dot² Pt)=-6`. This last average is an algebra diagnostic, not the correct average for a fixed-composition fluid. Covariance under simultaneous SO3 momentum/SU2 spin rotation, Hermiticity, parity, coupling scaling, isosinglet/triplet eigenvectors, spin singlet response, and Pauli cancellation for duplicate final states are also checked. No negative interference term is floored.

## Absolute elastic Born normalization and limits

For this **common-M nonrelativistic model**, the Schrödinger first-Born convention is `f_scatt=-mu_red A/(2π)`, `mu_red=M/2`. Thus the unpolarized differential cross section is

```
dsigma/dOmega = mu_red²/(4π²) <|A|²>spin,
```

in MeV^-2, converted to fm² using `(hbar c)²=(197.3269804MeV fm)²`. This is also obtained from the LO external-state conversion `|M_cov|²=(4M²)² <|A|²>` and `dσ/dΩ=|M_cov|²/(64π² s)` at **s=4M² at this order**. No exact relativistic amplitude or finite-p relativistic accuracy is implied by retaining exact s in part of that formula. This equality is a normalization identity, not an independent empirical cross-section anchor.

For np→np, use a definite outgoing neutron direction and integrate the full sphere once. For nn/pp the displayed differential is the labelled amplitude differential; the full sphere requires final identical-particle factor 1/2, or equivalently integrate one hemisphere. There is no initial 1/2 in a beam/target cross section. A same-species many-body collision **event rate** additionally requires initial unordered-pair counting, distinct from this cross section. A future MU rate must derive its own initial/final factorials with its labelled phase-space inventory; do not copy an overall factor from this NN example.

| pCM (MeV) | nn total Born (fm²) | np total Born (fm²) |
|---|---:|---:|
| 15 | 0.00610466732742 | 0.0193385191405 |
| 70 | 1.12198836295 | 3.08546799813 |
| 180 | 5.52161041628 | 12.3771185009 |

pp equals nn in this isospin-symmetric, Coulomb-free model. Separate 60/90-point angular rules agree within the 2e-12 gate; hemisphere and final factorial conventions agree. `(p/M)²` spans 0.000255–0.03675 here; this is a kinematic expansion diagnostic, not a bound on strong-interaction uncertainty. Contact terms, rescattering, finite energy transfer, effective spectra and correlations can matter even where numerical integration is extremely precise. These values are mathematical Born benchmarks, not NN phase-shift fits or physical neutron-star collision rates.

## Reproduction, failures and next gate

Run `python check_ope.py` in a fresh process using the existing campaign NumPy environment. There are **250/250 final author gates**; runtime is under one second. No external driver or other campaign source is imported or modified. `results.json` contains the finite-vector matrix/analytic forward values and cross sections; CHECKPOINT.json pins all local evidence. No derivative claims are made.

Preserved failures: the first validator had a missing whitespace SyntaxError and produced no numerical result. The next run passed236 gates and failed7 comparisons of an exactly forbidden charge amplitude against literal summation roundoff: a relative-error test against zero returned1 for cancellation residuals. Both source/raw versions are retained. The fixed validator keeps exact-zero matrix/analytic gates, compares literal forbidden cancellation in units of the natural coupling C, and leaves every finite-channel relative gate unchanged. The physical operator was not altered to repair these validation failures. The local PDF download failed DNS; web-accessed primary references are linked rather than inventing a downloaded source snapshot.

Before absolute MU: construct coherent weak insertion before/after strong scattering on each eligible line, with explicit neutron/proton isospin transitions, energy denominators from the same declared quasiparticle Hamiltonian, direct/exchange diagrams and a matched external-state normalization. Check any required interaction current for the chosen approximation. Off-pole kinematics and reciprocity must be tested independently before five-body phase integration. On-shell poles require a sourced matching/resummation prescription; no width is introduced here, and NWA/direct-width totals are not added to a separate MU term. The cold NMB cooling prescription and its qualifications remain unchanged. This elementary strong rung resolves the model operator without certifying SOM23's displayed Eq23 or reproducing its full MU rate.
