"""Selected all-field oracle in original momentum, with separate Gauss orders."""
import json, math
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
from nmbackend.constants import HBARC,M_E,M_MU
from nmbackend.reference.finite_t_leptons import state_mu,state_density,thermal_energy_density
OUT=Path(__file__).parent
rows=[]
def oracle(T,mu,m,order,tail):
    # p=sqrt(T*(T+m))*q is independent of the author's kinetic-energy rule.
    v,alpha=m/T,mu/T
    scale=math.sqrt(1+v)
    xend=math.sqrt((max(v,abs(alpha))+tail)**2-v*v)/scale
    nodes,weights=leggauss(order)
    q=(nodes+1)*xend/2; weights=weights*xend/2
    p=scale*q; E=np.hypot(p,v)
    kinetic=p*p/(E+v)
    # A common exponential factor is exact, not a classical approximation.
    logs=min(0.,abs(alpha)-v)
    a=np.exp(alpha-v-kinetic-logs)/(1+np.exp(alpha-v-kinetic))
    b=np.exp(-alpha-v-kinetic-logs)/(1+np.exp(-alpha-v-kinetic))
    wa=a*expit(E-alpha); wb=b*expit(E+alpha)
    measure=weights*scale**3*q*q
    total=measure@(a+b); net=measure@(a-b)
    energy=measure@(E*(a+b)); pressure=measure@(p*p/(3*E)*(a+b))
    a0=measure@(wa+wb)
    a1=measure@((E-alpha)*wa-(E+alpha)*wb)
    a2=measure@((E-alpha)**2*wa+(E+alpha)**2*wb)
    def restore(x,power):
        return math.copysign(math.exp(math.log(abs(x))+power*math.log(T)+logs-math.log(math.pi**2*HBARC**3)),x) if x else 0.
    return dict(net_density=restore(net,3),particle_density=restore(measure@a,3),
                antiparticle_density=restore(measure@b,3),energy_density=restore(energy,4),
                pressure=restore(pressure,4),entropy_density=restore(energy+pressure-alpha*net,3),
                free_energy_density=restore(alpha*net-pressure,4),
                cv_fixed_net_density=restore(a2-a1*a1/a0,3),
                dn_dmu_fixed_temperature=restore(a0,2),
                dn_dtemperature_fixed_mu=restore(a1,2),
                dmu_dtemperature_fixed_net_density=-a1/a0)
states=[(M_E/z,alpha*M_E/z,M_E) for z in (40.,100.,300.,600.) for alpha in (0.,.4,-.4)]
states += [(.73,-.29,M_E),(3.17,4.19,M_E),(13.7,91.3,M_MU),(.41,-117.2,M_MU)]
states += [(T,.37*T,.13*T) for T in (1e-30,1e-9,1e6)]
for T,mu,m in states:
    a=oracle(T,mu,m,384,80.); b=oracle(T,mu,m,512,100.)
    s=state_mu(T,mu,m); fine=state_mu(T,mu,m,rtol=5e-13,tail=90.)
    for name,value in b.items():
        actual=getattr(s,name); ref_a=a[name]
        # Symmetric odd moments vanish algebraically; do not divide a zero.
        if value==0:
            ok=actual==0; rel=0 if ok else None; oracle_error=0
        else:
            rel=abs(actual/value-1); oracle_error=abs(ref_a/value-1); ok=rel<3e-9 and oracle_error<3e-10
        rows.append(dict(T=T,mu=mu,m=m,field=name,actual=actual,reference=value,
                         relative_error=rel,oracle_refinement=oracle_error,
                         source_refinement=abs(getattr(fine,name)/actual-1) if actual else 0.,pass_=bool(ok)))
# Physical massive tiny roots and susceptibility are checked after forward.
for m,T in [(M_E,.83),(M_MU,17.3)]:
    for n in [-3e-29,3e-29]:
        s=state_density(T,n,m)
        linear=state_mu(T,0.,m).dn_dmu_fixed_temperature*s.chemical_potential
        rows.append(dict(field='massive_tiny_root',T=T,m=m,n=n,relative_error=abs(s.net_density/n-1),
                         linear_error=abs(linear/n-1),pass_=abs(s.net_density/n-1)<1e-9 and abs(linear/n-1)<1e-9))
result=dict(passed=sum(x['pass_'] for x in rows),failed=sum(not x['pass_'] for x in rows),rows=rows)
(OUT/'compat_momentum_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
assert not result['failed']
