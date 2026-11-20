"""Primal gates, analytic thermal-moment checks and first-order JAX validation."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from gm1_implicit_ad import *
from gm1_finite_t import gas
from stability import implicit_hessian

ROOT=Path(__file__).resolve().parent
start=time.monotonic();checks=[];states=[]
def check(name,actual,expected,tol,scale=1e-30):
    error=abs(actual-expected)/max(abs(expected),scale)
    checks.append(dict(name=name,actual=float(actual),expected=float(expected),scaled_error=float(error),tolerance=tol,passed=bool(error<tol)))

floors=np.array([1e-4]*8+[1e-6]*3+[1e-8])
directions=[np.array([.01,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,.02]),np.array([.012,1.7,-.015])]
for x in [np.array([.16,10.,.1]),np.array([.24,20.,.3]),np.array([.32,30.,.1])]:
    label=f'n={x[0]},T={x[1]},Y={x[2]}'
    y,J,diag=oracle_value_jacobian(x,return_diagnostics=True)
    s=state(*x);H,info=implicit_hessian(s)
    check(label+' local Hessian positive',float(np.linalg.eigvalsh(H)[0]>0),1.,1e-12)
    check(label+' primal equations',diag['max_equation_residual'],0.,2e-9,1.)
    check(label+' implicit linear equations',diag['implicit_linear_residual'],0.,2e-11,1.)
    actual=np.asarray(state_ad(jnp.array(x)))
    for k,name in enumerate(OUTPUTS):check(label+' primal '+name,actual[k],getattr(s,name),2e-11,1.)
    # Direct temperature derivative of each gas moment is an independent gate.
    tf=x[1]/HBARC;mass=s.dirac_mass_mev/HBARC
    for tag,nu in [('n',s.kinetic_mu_n_mev/HBARC),('p',s.kinetic_mu_p_mev/HBARC)]:
        r=thermal_response(tf,nu,mass);ht=.001*tf
        plus=gas(tf+ht,nu,mass);minus=gas(tf-ht,nu,mass)
        for key,field in [('n_T','n'),('rho_s_T','scalar'),('s_T','entropy')]:
            fd=(plus[field]-minus[field])/(2*ht)
            check(label+' analytic thermal moment '+tag+' '+key,r[key],fd,5e-6,1e-12)
    forward=np.asarray(jax.jacfwd(state_ad)(jnp.array(x)))
    reverse=np.asarray(jax.jacrev(state_ad)(jnp.array(x)))
    check(label+' jacfwd analytic Jacobian',np.linalg.norm(forward-J),0.,1e-12,max(np.linalg.norm(J),1.))
    check(label+' jacrev analytic Jacobian',np.linalg.norm(reverse-J),0.,1e-12,max(np.linalg.norm(J),1.))
    yref,Jref=oracle_value_jacobian(x,rtol=2e-12,tail=80.)
    for k,name in enumerate(OUTPUTS):
        scale=max(np.linalg.norm(Jref[k]),floors[k])
        check(label+' refined Jacobian '+name,np.linalg.norm(J[k]-Jref[k]),0.,3e-7,scale)
    # Exact thermodynamic/input-coordinate identities also check units/order.
    i={name:k for k,name in enumerate(OUTPUTS)}
    jf=J[i['free_energy_density']]
    check(label+' df/dT=-s',jf[1],-s.entropy_density,1e-12)
    check(label+' df/dn=mu_bar',jf[0],(1-x[2])*s.mu_n_mev+x[2]*s.mu_p_mev,1e-12)
    check(label+' df/dY=n*mu_diff',jf[2],x[0]*(s.mu_p_mev-s.mu_n_mev),1e-12)
    check(label+' Cv independent canonical formula',J[i['energy_density'],1],diag['canonical_cv_formula'],2e-11)
    check(label+' Phi_T canonical formula',J[i['phi_mev'],1],diag['canonical_dphi_dT_mev_per_mev'],2e-11)
    check(label+' Cv=T*ds/dT',J[i['energy_density'],1],x[1]*J[i['entropy_density'],1],1e-12)
    check(label+' frozen Cv positive',float(J[i['energy_density'],1]>0),1.,1e-12)
    # Density derivatives of chemical potentials must match the independently
    # qualified two-species implicit Hessian through dn_i/d(n,Y).
    density_map=np.array([[1-x[2],-x[0]],[x[2],x[0]]])
    jmu=J[[i['mu_n_mev'],i['mu_p_mev']]][:,[0,2]]
    check(label+' chemical Jacobian vs full Hessian',np.linalg.norm(jmu-H@density_map),0.,2e-11,max(np.linalg.norm(jmu),1.))
    ladders=[]
    for di,d in enumerate(directions):
        predicted=J@d
        _,jvp=jax.jvp(state_ad,(jnp.array(x),),(jnp.array(d),))
        check(label+f' direction{di} JVP',np.linalg.norm(np.asarray(jvp)-predicted),0.,1e-12,max(np.linalg.norm(predicted),1.))
        fds=[];residuals=[]
        for h in (.02,.01,.005):
            plus=state(*(x+h*d));minus=state(*(x-h*d))
            residuals.extend([plus.max_normalized_equation_residual,minus.max_normalized_equation_residual])
            yp=np.array([getattr(plus,name) for name in OUTPUTS]);ym=np.array([getattr(minus,name) for name in OUTPUTS])
            fds.append((yp-ym)/(2*h))
        rich=(4*fds[-1]-fds[-2])/3
        for k,name in enumerate(OUTPUTS):check(label+f' direction{di} FD '+name,predicted[k],rich[k],3e-5,floors[k])
        check(label+f' direction{di} perturbation residuals',max(residuals),0.,2e-9,1.)
        ladders.append(dict(direction=d.tolist(),steps=[.02,.01,.005],finite_differences=[fd.tolist() for fd in fds],richardson=rich.tolist(),analytic=predicted.tolist(),max_perturbed_equation_residual=max(residuals)))
    states.append(dict(x=x.tolist(),outputs=y.tolist(),jacobian=J.tolist(),diagnostics=diag,directional_ladders=ladders))

# JIT primal/JVP and reverse scalar functional at an already primal-qualified state.
x=jnp.array([.16,10.,.1]);d=jnp.array([.012,1.7,-.015])
y,J=oracle_value_jacobian(np.asarray(x))
jit_value=np.asarray(jax.jit(state_ad)(x))
check('JIT primal',np.linalg.norm(jit_value-y),0.,2e-11,max(np.linalg.norm(y),1.))
jit_jvp=jax.jit(lambda z,v:jax.jvp(state_ad,(z,),(v,))[1])(x,d)
check('JIT first JVP',np.linalg.norm(np.asarray(jit_jvp)-J@np.asarray(d)),0.,2e-12,max(np.linalg.norm(J@np.asarray(d)),1.))
weights=jnp.array([.01,.01,.01,.01,.01,.01,.01,.01,1.,1.,1.,10.])
grad=jax.grad(lambda z:jnp.dot(weights,state_ad(z)))(x)
check('reverse weighted scalar functional',np.linalg.norm(np.asarray(grad)-np.asarray(weights)@J),0.,2e-12,max(np.linalg.norm(np.asarray(weights)@J),1.))
try:
    jax.jacfwd(jax.jacrev(state_ad))(x)
except (ValueError,NotImplementedError) as exc:
    second=dict(supported=False,error_type=type(exc).__name__,message=str(exc).split('\n')[0])
else:
    second=dict(supported=True,error_type=None,message='Unexpected success: inspect derivative semantics')
check('higher derivatives explicitly unsupported',float(not second['supported']),1.,1e-12)
# Explicit staged-domain failure contract, without callback exceptions.
bad=jnp.array([.16,0.,.1])
check('invalid staged state returns NaN sentinel',float(np.all(np.isnan(np.asarray(jax.jit(state_ad)(bad))))),1.,1e-12)
_,bad_jvp=jax.jvp(state_ad,(bad,),(d,))
check('invalid staged JVP returns NaN sentinel',float(np.all(np.isnan(np.asarray(bad_jvp)))),1.,1e-12)
try:
    oracle_value_jacobian(np.asarray(bad))
except ValueError:
    strict_rejected=True
else:
    strict_rejected=False
check('strict host preflight rejects invalid T',float(strict_rejected),1.,1e-12)
report=dict(passed=all(c['passed'] for c in checks),checks=checks,states=states,output_order=OUTPUTS,
            second_derivative_probe=second,runtime_seconds=time.monotonic()-start,
            environment=dict(python=sys.version,executable=sys.executable,jax=jax.__version__,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),devices=[str(d) for d in jax.devices()]),
            frozen_sha256={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT.parent/'rmf_reference/gm1_finite_t.py',ROOT.parent/'rmf_phase/stability.py']})
(ROOT/'validation_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(f"{sum(c['passed'] for c in checks)}/{len(checks)} checks pass in {report['runtime_seconds']:.2f}s")
print('Second derivative:',second)
for c in checks:
    if not c['passed']:print('FAIL',c)
raise SystemExit(0 if report['passed'] else 1)
