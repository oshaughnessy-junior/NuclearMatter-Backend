"""Independent positive 1D quadrature for five fixed-length vector closure."""
import math
from scipy.integrate import quad

def angular(lengths,*,epsrel=2e-12):
    """Includes all five dOmega and independent neutrino dOmega=4pi.

    For normalized orientations, rho2(R)=1/(8pi*a*b*R).
    rho3(R)=overlap_length/(16pi*c*d*e*R). Integrate their
    product over d^3R, then multiply by (4pi)^6.
    """
    if len(lengths)!=5 or not all(math.isfinite(x) and x>0 for x in lengths):raise ValueError('five positive lengths required')
    a,b,c,d,e=lengths
    lo,hi=abs(a-b),a+b;ql,qh=abs(c-d),c+d
    def overlap(R):return max(0.,min(qh,R+e)-max(ql,abs(R-e)))
    points=sorted({lo,hi,*[x for x in [e,ql-e,ql+e,e-ql,qh-e,qh+e,e-qh] if lo<x<hi]})
    total=0.;error=0.
    for l,h in zip(points[:-1],points[1:]):
        q,er=quad(overlap,l,h,epsabs=1e-14,epsrel=epsrel);total+=q;error+=er
    pref=128*math.pi**5/math.prod(lengths)
    return dict(angular=pref*total,estimated_error=pref*error,overlap_area=total,panels=points)
