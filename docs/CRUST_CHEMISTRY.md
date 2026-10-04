# Crust thermal chemistry for native cooling

`physics.crust.CrustTable` reads NSCool's six-column crust table outside JIT.
`crust_chemistry` evaluates it in JAX, including the virtual top row matched
through the supplied core density, baryon density, and electron fraction.
The evaluator follows `get_crust_chemistry` in NSCool `precool.f`: linear
interpolation in mass density, piecewise ion species, and intervals
`lower < rho <= upper`. Invalid queries or incompatible junctions return NaNs.

The returned `n` is the baryon density prescribed for thermal chemistry.
It does not replace the native EOS baryon density used for stellar geometry.
Combining a native EOS with this crust table is an explicit hybrid prescription;
thermodynamic consistency has not been established. Likewise, `v_ion` reproduces
NSCool's mass-density prescription, rather than deriving a volume fraction
from the native EOS or chemical number density.

Derivatives include motion of the virtual endpoint and its electron-fraction
matching. They are conditional between species contacts and changes in which
table rows remain below the core boundary. No smooth derivative is claimed
across those contacts. Before integrating a whole thermal grid, measure the
chemical/native baryon-density mismatch throughout the crust and at the junction.
This module alone does not close C3 or provide a whole-star cooling curve.

Validation uses an independent analytic moving-endpoint Jacobian, forward and
reverse AD, centered finite differences at three steps, species-contact and
invalid-domain checks, and chemistry parity with actual unpaired NSCool dumps
for 1.4 and 2.0 solar masses. The five focused tests pass with NSCool required.
The first run exposed trailing table annotations and an unphysical synthetic
contact fixture; both were corrected while preserving the physical validity guard.
