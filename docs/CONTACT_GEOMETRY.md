# Contact geometry for native thermal integration

`physics.geometry` inverts the actual Hermite energy density used by the native
TOV solver and maps supported enthalpies to enclosed `r^3`. Its implicit root
rule differentiates both the target energy and EOS nodes. It rejects unsupported
or nonmonotone intervals, nearly tangent roots, and excessive root residuals.
The scalar energy input uses geometric units, km^-2.

Geometry interpolates radius in the square root of central enthalpy drop and
enclosed mass in `r^3`, retaining their central scaling. The lapse follows the
enthalpy first integral. The caller must supply the central enthalpy that
produced the saved profile; the validity checks cannot certify that relationship.
EOS support and support inside this star are separate requirements.

`volume_quadrature` supplies Gauss nodes and proper-volume weights for ordered
contact edges in km^3. Equal edges retain finite geometry and zero weights.
Moving edges remain in the differentiated calculation, retaining interface
motion without smoothing rate jumps. Invalid edges give NaN weights. Thermal
redshift factors belong to the caller and are not included in these weights.
Derivatives are conditional on fixed contact topology and interpolation branches.

Six focused tests check independent analytic roots and sensitivities, invalid
Hermite intervals, central scaling, uniform-density Schwarzschild proper volume,
a moving two-rate interface, zero-width segments and stellar coverage. The
Schwarzschild test's enthalpy profile is synthetic; it does not validate a full
TOV/EOS/lapse solution. Contact and rate-amplitude derivatives have independent
analytic oracles; mass/radius derivative agreement is method consistency only.

The high-order analytic convergence check does not certify default order eight
for real thermal rates. Refine quadrature and saved-profile resolution together.
Native thermal parity, chemistry interfaces, conducting blanket, full cooling
curves and C3 gradient gates remain separate work.
