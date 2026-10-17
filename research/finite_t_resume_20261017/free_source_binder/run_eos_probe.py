"""Fresh independent momentum forward anchors, then response step ladders."""
from pathlib import Path
from dataclasses import asdict
import json,math,time
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import expit
from nmbackend.reference import free_matter as F
P=Path(__file__).resolve().parent

def momentum(T,mu,m,pairs):
    end=max(m,abs(mu))+65*T;upper=math.sqrt((end-m)*(end+m))
    def integrand(p):
        E=math.hypot(p,m);x=(E-mu)/T;y=(E+mu)/T
        a=expit(-x);b=expit(-y) if pairs else 0.
        h=lambda z:np.logaddexp(0.,-abs(z))+abs(z)*expit(-abs(z))
        entropy=h(x)+(h(y) if pairs else 0.)
        measure=p*p/(math.pi**2*F.HC**3)
        return measure*np.array([a-b,a,b,E*(a+b),p*p/(3*E)*(a+b),entropy])
    value,error=quad_vec(integrand,0.,upper,epsabs=1e-22,epsrel=2e-12,limit=300)
    return dict(zip(['net','particle','anti','energy','pressure','entropy'],map(float,value)))

def main():
    start=time.monotonic();checks=[];rows=[]
    def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and abs(error)<=tol)))
    for T,n,y,photons in [(2.,1e-7,.3,'absent'),(2.03,1.05e-7,.34,'trapped_LTE'),(1.7,2e-7,.21,'absent')]:
        s=F.state_density(T,n,y,photons=photons);anchors={}
        # Unperturbed momentum forward checks BEFORE any finite-difference response.
        for name,component,pairs in [('n',s.neutron,False),('p',s.proton,False),('e',s.electron,True)]:
            a=momentum(T,component.chemical_potential,component.mass,pairs);anchors[name]=a
            number=component.net_density if pairs else component.particle_density
            for key,target in [('net',number),('energy',component.energy_density),('pressure',component.pressure),('entropy',component.entropy_density)]:ck('forward momentum '+str((T,name,key)),a[key]/target-1,2e-9)
        ck('actual charge closure '+str(T),s.electron.net_density/s.proton.particle_density-1,2e-9)
        ck('electron pairs actual '+str(T),(anchors['e']['particle']-anchors['e']['anti'])/s.electron.net_density-1,2e-9)
        ck('total free energy '+str(T),(s.free_energy_density-(s.neutron.free_energy_density+s.proton.free_energy_density+s.electron.free_energy_density-s.photon_pressure))/s.energy_density,2e-12)
        ladders=[]
        for eps in [.002,.001,.0005]:
            a=F.state_density(T+eps,n,y,photons=photons);b=F.state_density(T-eps,n,y,photons=photons);cv=(a.energy_density-b.energy_density)/(2*eps)
            a=F.state_density(T,n*(1+eps),y,photons=photons);b=F.state_density(T,n*(1-eps),y,photons=photons);en=(a.energy_density-b.energy_density)/(2*n*eps)
            a=F.state_density(T,n,y+eps,photons=photons);b=F.state_density(T,n,y-eps,photons=photons);ey=(a.energy_density-b.energy_density)/(2*eps)
            for name,num,target in [('Cv',cv,s.cv_fixed_nB_Ye),('enB',en,s.denergy_dnB_fixed_T_Ye),('eY',ey,s.denergy_dYe_fixed_nB_T)]:ck('density-root response ladder '+str((T,eps,name)),num/target-1,1e-4)
            ladders.append(dict(step=eps,Cv=cv,enB=en,eY=ey))
        rows.append(dict(state=asdict(s),identity=s.identity,momentum=anchors,ladders=ladders))
    result=dict(checks=checks,passed=sum(c['passed'] for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,seconds=time.monotonic()-start,attribution='Actual staged package state factories; independent momentum integrals precede FD responses; no weak source evaluation')
    (P/'eos_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['passed','failed','seconds']}))
    return len(result['failed'])

if __name__=='__main__':raise SystemExit(main())
