# Smooth steady conducting layer

`physics.blanket._steady_layer` is an internal building block for constant-flux transport in prescribed spherical geometry. It integrates redshifted temperature inward using an implicit midpoint step in log temperature. The scalar root has an implicit JAX derivative; the solve does not differentiate a fixed number of Picard iterations.

Inputs are descending radii in km, nonnegative luminosity at infinity in erg/s, and positive boundary temperature at infinity in K. The geometry callback returns lapse and compactness `2m/r`. The conductivity callback receives local temperature and radius, and returns conductivity in erg cm^-1 s^-1 K^-1. The output temperature is redshifted.

The step assumes a smooth conductivity branch and a unique non-tangent root. Finite bracket expansion and bisection locate that root. Invalid input, geometry, brackets, conductivity, or residuals give NaN temperature and false step validity. Diagnostics expose residual, root slope, conductivity, and validity; residual divided by absolute slope estimates local root conditioning. These guards do not prove root uniqueness or qualify a phase interface.

Independent flat-space, prescribed Schwarzschild, and temperature-dependent conductivity solutions check values, gradients, and grid refinement. Zero flux preserves constant redshifted temperature while retaining the correct derivative with respect to absolute luminosity. The Schwarzschild fixture is a mathematical transport geometry, not a vacuum crust model.

This layer neglects heat storage, internal heating, and neutrino losses. In the current hot unpaired reference, total crust neutrino luminosity exceeds photon luminosity by about 2.44. Constant flux through that entire crust is therefore an approximation/parity calculation, not demonstrated physical transport fidelity. Cooler domains need evaluated budgets and relaxation checks.

Actual crust conductivity also switches at Gamma=210. This helper does not locate that moving interface. Native crust transport, EOS gradients, and the whole cooling-curve gate remain open; callers must qualify branch occupancy and refine their radial grid before interpreting derivatives.
