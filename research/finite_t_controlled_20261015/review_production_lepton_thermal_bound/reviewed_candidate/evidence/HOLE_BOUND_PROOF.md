# Dimensional positive-hole omission bound

For the same represented positive T, v=m/T and xc used in the quadrature,
let C=1/[pi² HBARC³]. The physical hole energy is

H = C T^4 ∫[0,xc] sqrt[x(x+2v)] (v+x) (xc−x) vacancy(x) dx.

All factors are nonnegative, vacancy<=1, and geometry is monotonic for
v>=0. Thus

H <= C T^4 sqrt[xc(xc+2v)] (v+xc) xc²/2.

C carries MeV^-3 fm^-3, T^4 carries MeV^4, and all remaining factors are
dimensionless, so this bounds MeV fm^-3, the returned thermal-energy unit.
Ignoring a vacancy smaller than1 only makes the bound more conservative.

The implementation avoids a floating logarithm at the admission threshold.
For each positive represented x, frexp gives an integer e with x<2^e.
Let ec,et,em be the exponents for xc,T,C. If v>0, let ev be its exponent,
ea=max(ec,ev+1)+1 and eb=max(ec,ev)+1. Then xc+2v<2^ea and xc+v<2^eb.
For v=0, ea=eb=ec suffices. Consequently

H < 2^B,
B = 4et + em + (ec+ea)/2 + eb + 2ec − 1.

These integers/half-integers and their sums are exactly represented in
float64: frexp exponents for finite float64 factors are bounded near±1074,
far below the integer exactness limit. No exponentiation or approximate
natural log is used for the decision. For round-to-nearest float64 the
smallest subnormal is2^-1074; any positive value strictly below2^-1075
rounds to zero. Therefore omitting H only when B<=−1075 is conservative,
including equality because the upper power-of-two inequalities are strict.
The function checks xc>0 first; xc=0 has an exactly empty hole interval.

This is a bound for a positive physical COMPONENT, not total heat or net
number, and is not a density/temperature floor. If the bound fails this gate,
the existing hole quadrature and all its unresolved-status guards remain.
Excitations and antiparticles are always independently integrated as before.
No zero quadrature is accepted merely because its sampled values are zero.

At the two resolved electron states m/T600,mu/T±.53, the independent tighter
analytic bound from the preceding review is log(H)<=−1037.143 MeV fm^-3.
The dyadic bound is looser but still below the half-subnormal limit; exact
values are in hole_preservation_results.json. Total thermal energy is normal,
approximately2.41148076418e−274 MeV fm^-3, and is never set to zero.

Tests check both sides of the exponent gate with dyadic massless controls,
an independent exact polynomial integral with a nonzero represented hole,
and retained unresolved-quadrature rejection. These selected tests accompany
the analytic bound; they do not certify a uniform extreme-domain rectangle,
relative accuracy of subnormal totals or differentiation through quadrature.
