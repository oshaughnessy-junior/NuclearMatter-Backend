# Exact equations and the port's numerical changes

For spin-two particles, C=1/[pi² HBARC³], x=(E−m)/T,v=m/T,
eta=(mu−m)/T and g=max(1,v,eta). The physical measure is
C*T³*pbar*Ebar dx, where pbar=sqrt[x(x+2v)] and Ebar=v+x.
The dimensionless geometry in the code is pbar*Ebar/g². Multiplying its
integral by C*T³*g² restores the exact measure; it is not an EOS approximation.

Each occupation is divided by f0=expit(eta). For eta<=0 its ratio is
exp(−x)*(1+exp(eta))/(1+exp(eta−x)), avoiding subtraction of huge logarithms.
For eta>0 the direct ratio of expit values is resolved. A common logarithmic
prefactor combines C,T³,g² and f0 before ordinary output restoration.
Dimensionless energy/response scales balance the vector quadrature's max norm.
All adaptive calls use epsabs=0, retained error/success guards and explicit
Fermi-scale panels; no amplitude floor or classical replacement is applied.

K directly integrates T*x*f. P directly integrates
T*x*(x+2v)/[3*(v+x)]*f. Total energy m*n+K retains rest energy. Entropy
integrates the positive occupation entropy with a stable dilute scale;
F=mu*n−P is the equivalent ideal-gas free energy identity. These are not
subtractions of two large rest-dominated values to obtain K or entropy.

Let W denote the actual measure times f(1−f), A0=∫W and xbar=∫xW/A0.
Then dn/dmu=A0/T, dn/dT=∫(x−eta)W/T and mu_T|n=−∫(x−eta)W/A0.
The covariance identity gives
Cv|n = [∫(E−mu)²W−(∫(E−mu)W)²/A0]/T²
      = ∫(x−xbar)²W.
The implementation uses the latter positive second integral. This avoids
catastrophic subtraction in dilution/rest-dominated second moments while
retaining the same fixed-mass thermodynamic function. A nonpositive or
unrepresented n,chi,Cv still rejects, rather than inserting a heat capacity.

The density root uses the monotone logarithmic density against log(n_target),
so tiny positive n does not become zero under an absolute residual tolerance.
Its initial NR estimate is only a bracket guess. Returning total mu=m+T*eta
can quantize away a tiny offset; an explicit ordinary forward reconstruction
to1e−9 relative is mandatory afterward. A tested unrepresented-offset case
rejects. Internal logs do not create a public log-density API or a relative
precision claim for subnormal/unrepresented fields.

Every response holds mass fixed. Parameterizing separate ideal-gas calls by
a user-supplied positive constant mass is not a thermodynamically complete
n-/T-dependent effective-mass EOS: no mass derivatives, rearrangement,
background energy, field potentials or weak-current mass changes are supplied.
Likewise the omitted antiparticle gas is a declared model exclusion, not a
uniform thermal negligibility certificate or a hidden runtime gate.
