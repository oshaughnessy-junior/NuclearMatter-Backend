"""Controlled finite-T charged-current phase-space threshold, no imposed width.

Local constant offsets are calibrated once to exact T=0 beta equilibrium for
each chosen composition, then held fixed. This is not an interacting EOS.
"""
from pathlib import Path
import sys,math,json,hashlib,time
import numpy as np
from scipy.integrate import quad
from scipy.special import expit,log_expit,logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
for sub in ['nr_baryon_eos','capture_endpoint','quantum_slab']:
    sys.path.insert(0,str(ROOT/sub))
import free_nucleons as eos
import guarded_capture
import mapped_capture
import allowed_capture as a
from free_density_response import log_response

def cold_reference(nB,Yp):
    nn,nP=nB*(1-Yp),nB*Yp
    n=eos.cold_state(nn);p=eos.cold_state(nP)
    mue=math.hypot(p['pF'],a.ME)
    delta=p['EF']+mue-n['EF']
    return dict(nB=nB,Yp=Yp,offset_neutron=delta,offset_proton=0.,
                pFn=n['pF'],pFp=p['pF'],pFe=p['pF'],delta_p=n['pF']-2*p['pF'],
                EF_n=n['EF'],EF_p=p['EF'],mu_e_cold=mue,
                cold_beta_residual=delta+n['EF']-p['EF']-mue,
                calibration_temperature=0.)

def make_state(reference,T):
    n,Y=reference['nB'],reference['Yp']
    return eos.matter_state(T,n*(1-Y),n*Y,offset_neutron=reference['offset_neutron'],offset_proton=0.)

def independent_charge_density(matter):
    """Scalar momentum integrals, independent of both density root integrators."""
    out=[]
    for sp in [matter.neutron,matter.proton]:
        kmax=max(sp.kinetic_mu,0)+80*sp.T
        end=math.sqrt(2*sp.mass*kmax)
        points=[math.sqrt(2*sp.mass*k) for k in [sp.kinetic_mu-20*sp.T,sp.kinetic_mu,sp.kinetic_mu+20*sp.T] if 0<k<kmax]
        value=quad(lambda p:p*p*expit((sp.kinetic_mu-p*p/(2*sp.mass))/sp.T),0,end,points=points,epsabs=1e-12,epsrel=2e-11,limit=250)[0]/(math.pi**2*eos.HBARC**3)
        out.append(value)
    el=matter.electron;mu=el.chemical_potential;T=el.temperature;m=el.mass
    endE=max(abs(mu),m)+80*T;end=math.sqrt(endE*endE-m*m)
    points=[math.sqrt(E*E-m*m) for E in [mu-20*T,mu,mu+20*T] if m<E<endE]
    value=quad(lambda p:p*p*(expit((mu-math.hypot(p,m))/T)-expit((-mu-math.hypot(p,m))/T)),0,end,points=points,epsabs=1e-12,epsrel=2e-11,limit=250)[0]/(math.pi**2*eos.HBARC**3)
    out.append(value)
    return out

def independent_log_chi(state,Enu,angle_order=64):
    """Adaptive Ee near electron Fermi surface with independent q angular rule.

    Integrates y=(Ee-mu_e)/T and factors a measured common log scale. It shares
    the established response equation, not optical energy panels/quadrature.
    """
    s=state.baryons;T=s.T;mu=state.mu_lepton
    qs,qws=np.polynomial.legendre.leggauss(angle_order)
    # y-panel edges resolve Fermi occupations and extend well beyond selected
    # below-threshold activation scales. No cold-support cutoff is imposed.
    low=(a.ME-mu)/T
    upper=100.
    edges=sorted({low,upper}|{v for v in [-80.,-40.,-20.,-8.,0.,8.,20.,40.,80.] if low<v<upper})
    def log_integrand(y):
        Ee=mu+T*y
        if Ee<=a.ME:return -math.inf
        p=math.sqrt((Ee-a.ME)*(Ee+a.ME))
        # q-coordinate nodes cover exactly |Enu-pe|..Enu+pe.
        qlo,qhi=abs(Enu-p),Enu+p
        q=qlo+.5*(qs+1)*(qhi-qlo);qw=.5*(qhi-qlo)*qws
        terms=[]
        for qq,ww in zip(q,qw):
            c=(Enu*Enu+p*p-qq*qq)/(2*Enu*p)
            c=max(-1.,min(1.,float(c)))
            terms.append(math.log(float(ww*Ee*qq/Enu*a.angular_factor(Ee,c)))+log_response(s,float(qq),Enu-Ee))
        return math.log(a.K*T)+float(log_expit((Ee-mu)/T))+float(logsumexp(terms))
    probes=[v for v in [-80.,-40.,-20.,-8.,0.,8.,20.,40.,80.] if low<v<upper]
    offset=max(log_integrand(v) for v in probes)
    value=0.;error=0.
    for lo,hi in zip(edges[:-1],edges[1:]):
        v,e=quad(lambda y:math.exp(log_integrand(y)-offset),lo,hi,epsabs=1e-13,epsrel=2e-10,limit=250)
        value+=v;error+=e
    return offset+math.log(value),error/value

def main():
    start=time.monotonic();rows=[];checks=[];failures=[]
    paths=[Path(eos.__file__),Path(guarded_capture.__file__),Path(mapped_capture.__file__),Path(a.__file__),ROOT/'response_reference/free_density_response.py',ROOT/'thermodynamics/lepton_reference.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def gate(name,error,tolerance):
        rec=dict(name=name,error=float(error),tolerance=tolerance,passed=bool(math.isfinite(error) and error<tolerance));checks.append(rec)
        if not rec['passed']:failures.append(rec)
    for Y in [.10,1/9,.12]:
        ref=cold_reference(.01,Y)
        gate(f'cold_beta_Y{Y}',abs(ref['cold_beta_residual']),1e-12)
        for T in [.01,.03,.1,.3]:
            matter=make_state(ref,T);opt=matter.optical('nue');Enu=3*T
            independent=independent_charge_density(matter)
            targets=[.01*(1-Y),.01*Y,.01*Y]
            gate(f'independent_density_Y{Y}_T{T}',float(np.max(abs(np.asarray(independent)/targets-1))),1e-8)
            try:
                coarse=guarded_capture.log_coefficients(opt,Enu,energy_order=24,angle_order=24)
                fine=guarded_capture.log_coefficients(opt,Enu,energy_order=48,angle_order=48)
                refined=guarded_capture.log_coefficients(opt,Enu,energy_order=72,angle_order=72,tail=60.)
                gate(f'capture_order_Y{Y}_T{T}',max(abs(math.expm1(fine[i]-refined[i])) for i in range(2)),3e-6)
                gate(f'capture_KMS_Y{Y}_T{T}',abs(refined[0]-refined[1]+(Enu-opt.mu_neutrino)/T),3e-10)
                # Independent coordinate comparison at every selected state.
                other,err=independent_log_chi(opt,Enu)
                gate(f'independent_capture_Y{Y}_T{T}',abs(math.expm1(refined[1]-other)),3e-6)
                gate(f'independent_quad_error_Y{Y}_T{T}',err,3e-7)
                rows.append(dict(reference=ref,T=T,Enu=Enu,Enu_over_T=3.,affinity_mu_e_plus_mu_p_minus_mu_n=opt.mu_neutrino,
                                 mu_n=matter.neutron.mu,mu_p=matter.proton.mu,mu_e=matter.electron.chemical_potential,
                                 independent_densities=independent,log_j=refined[0],log_chi=refined[1],
                                 log_j_over_T2=refined[0]-2*math.log(T),log_chi_over_T2=refined[1]-2*math.log(T),
                                 independent_log_chi=other,coarse_logs=coarse[:2],fine_logs=fine[:2],
                                 discarded_panels=refined[2],pFn_over_M=matter.neutron.fermi_momentum_over_mass,
                                 pFp_over_M=matter.proton.fermi_momentum_over_mass,T_over_EFp=T/ref['EF_p']))
            except (ValueError,ArithmeticError,RuntimeError) as exc:
                failure=dict(Y=Y,T=T,kind='integration exception',error=repr(exc));failures.append(failure)
    final={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report=dict(status='PASS' if not failures and hashes==final else 'FAIL',checks=checks,failures=failures,rows=rows,source_sha256=hashes,end_source_sha256=final,runtime_seconds=time.monotonic()-start,
                scope='selected electron capture/inverse spectral coefficients in frozen local constant-offset commonM NR quasiparticle model; not full Urca emissivity or interacting EOS')
    path=Path(__file__).with_name('results.json');path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=len(checks),rows=len(rows),failures=failures,runtime=report['runtime_seconds']),indent=2))

if __name__=='__main__':main()
