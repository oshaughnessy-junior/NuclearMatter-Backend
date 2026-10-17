"""Coupling-work identities, pinned-source partials and re-solved FD ladders."""
from __future__ import annotations
import ast
import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import scipy
from gm1_parameter_ad import *
from gm1_implicit_ad import oracle_value_jacobian as state_only_oracle

ROOT=Path(__file__).resolve().parent
start=time.monotonic();checks=[];reports=[]
def check(name,a,b,tol,scale=1e-30):
    error=abs(a-b)/max(abs(b),scale)
    checks.append(dict(name=name,actual=float(a),expected=float(b),scaled_error=float(error),tolerance=tol,passed=bool(error<tol)))

def source_namespace():
    source=ast.parse((ROOT.parent/'rmf_reference/source_snapshot/rmft.py').read_text())
    names={'_kf','_rho_s_one','_eps_kin_one','_vector_fields','_eps_p_from_state'}
    ns=dict(jnp=np,math=math,_PI2=math.pi**2,HBARC=HBARC)
    exec(compile(ast.Module(body=[v for v in source.body if isinstance(v,ast.FunctionDef) and v.name in names],type_ignores=[]),'pinned_NMB_source','exec'),ns)
    return ns
ns=source_namespace();floor=np.array([1e-5]*8+[1e-7]*3+[1e-9])
for n,T,Y in [(.16,10.,.1),(.24,20.,.3),(.32,30.,.1)]:
    x=pack_state(n,T,Y);label=f'n={n},T={T},Y={Y}'
    y,J,diag=oracle_value_jacobian(x,return_diagnostics=True)
    old_y,old_J=state_only_oracle(x[:3])
    check(label+' baseline frozen primal',np.linalg.norm(y-old_y),0.,2e-11,max(np.linalg.norm(old_y),1.))
    for i,k in enumerate(OUTPUTS):check(label+' unchanged state Jacobian '+k,np.linalg.norm(J[i,:3]-old_J[i]),0.,2e-11,max(np.linalg.norm(old_J[i]),floor[i]))
    check(label+' baseline roots',diag['max_equation_residual'],0.,2e-9,1.)
    check(label+' local stable Hessian',float(min(diag['chemical_hessian_eigenvalues_mev_fm3'])>0),1.,1e-12)
    check(label+' implicit linear residual',diag['implicit_linear_residual'],0.,2e-10,1.)
    forward=np.asarray(jax.jacfwd(state_parameter_ad)(jnp.array(x)));reverse=np.asarray(jax.jacrev(state_parameter_ad)(jnp.array(x)))
    check(label+' JAX jacfwd',np.linalg.norm(forward-J),0.,2e-12,max(np.linalg.norm(J),1.))
    check(label+' JAX jacrev',np.linalg.norm(reverse-J),0.,2e-12,max(np.linalg.norm(J),1.))
    yp,Jp=oracle_value_jacobian(x,rtol=2e-12,tail=80.)
    for j,name in enumerate(PARAMETERS,3):
        check(label+' refined coupling '+name,np.linalg.norm(J[:,j]-Jp[:,j]),0.,3e-7,max(np.linalg.norm(Jp[:,j]),1.))
    # Independent source-defined explicit partials: hold Phi and density fixed,
    # eliminate vectors algebraically exactly as the actual pinned source does.
    phi=y[0]/HBARC;source_partials=[]
    for j,key in enumerate(PARAMETERS,3):
        source_steps=[]
        for frac in (.002,.001,.0005):
            h=frac*abs(x[j]);pm=dict(GM1);pp=dict(GM1);pm[key]-=h;pp[key]+=h
            em,_,_=ns['_eps_p_from_state'](n,Y,phi,pm,leptons=False)
            ep,_,_=ns['_eps_p_from_state'](n,Y,phi,pp,leptons=False)
            source_steps.append((ep-em)/(2*h))
        source_partial=(4*source_steps[-1]-source_steps[-2])/3
        source_partials.append(source_partial)
        check(label+' source partial Hellmann-Feynman '+key,J[10,j],source_partial,3e-8,1e-8)
    # Mixed Maxwell relations: differentiate the explicit HF formulas with
    # respect to state using the independently frozen state-only Phi tangent.
    dphi=old_J[0]/HBARC;n3=n*(2*Y-1)
    hf_state=np.vstack([-HBARC*phi/(GM1['C_sigma']**2)*dphi,
        HBARC*np.array([n,0.,0.]),
        .25*HBARC*n3*np.array([2*Y-1,0.,2*n]),
        HBARC*(939./HBARC)*phi*phi*dphi,HBARC*phi**3*dphi])
    for pindex,key in enumerate(PARAMETERS):
        col=pindex+3
        check(label+' mixed HF Maxwell entropy '+key,-hf_state[pindex,1],J[11,col],3e-10,1e-10)
        check(label+' mixed HF Maxwell density '+key,hf_state[pindex,0],(1-Y)*J[6,col]+Y*J[7,col],3e-10,1e-8)
        check(label+' mixed HF Maxwell composition '+key,hf_state[pindex,2],n*(J[7,col]-J[6,col]),3e-10,1e-8)
    ladders=[]
    for j,key in enumerate(PARAMETERS,3):
        fds=[];gates=[]
        for frac in (.002,.001,.0005):
            h=frac*abs(x[j]);xp=x.copy();xm=x.copy();xp[j]+=h;xm[j]-=h
            plus,_,dp=oracle_value_jacobian(xp,return_diagnostics=True)
            minus,_,dm=oracle_value_jacobian(xm,return_diagnostics=True)
            # Forward branch/condition gates precede derivative comparisons.
            maxres=max(dp['max_equation_residual'],dm['max_equation_residual'])
            mineig=min(dp['chemical_hessian_eigenvalues_mev_fm3']+dm['chemical_hessian_eigenvalues_mev_fm3'])
            check(label+f' perturbed roots {key} step{frac}',maxres,0.,2e-9,1.)
            check(label+f' perturbed local stability {key} step{frac}',float(mineig>0),1.,1e-12)
            fds.append((plus-minus)/(2*h));gates.append(dict(relative_step=frac,absolute_step=h,max_root_residual=maxres,min_hessian_eigenvalue=mineig))
        rich=(4*fds[-1]-fds[-2])/3
        for i,name in enumerate(OUTPUTS):check(label+' coupling FD '+key+' '+name,J[i,j],rich[i],5e-5,floor[i])
        hf=diag['hellmann_feynman_free_energy_derivatives'][j-3]
        check(label+' Hellmann-Feynman '+key,J[10,j],hf,2e-12)
        check(label+' coupling first-law '+key,J[8,j]-T*J[11,j],hf,2e-11)
        # This is intentionally not the fixed-parameter Gibbs-Duhem identity.
        pressure_work=n*(1-Y)*J[6,j]+n*Y*J[7,j]-J[9,j]
        check(label+' parameter pressure work '+key,pressure_work,hf,2e-11)
        ladders.append(dict(parameter=key,absolute_input=x[j],gates=gates,finite_differences=[v.tolist() for v in fds],richardson=rich.tolist(),analytic=J[:,j].tolist(),source_fixed_scalar_partial=source_partials[j-3]))
    # Vector-coupling exact checks expose the missing-work factor-of-two bug.
    check(label+' C_omega exact pressure derivative',J[9,4],HBARC*n*n/2,2e-12)
    check(label+' C_rho exact pressure derivative',J[9,5],HBARC*(n*(2*Y-1))**2/8,2e-12)
    for col,key in [(4,'C_omega'),(5,'C_rho')]:
        check(label+' naive pressure factor two '+key,diag['naive_fixed_parameter_pressure_derivatives'][col-3],2*J[9,col],2e-12)
        check(label+' vector coupling no entropy shift '+key,J[11,col],0.,1e-12,1.)
    d=np.array([.004,.7,.003,.03,.02,.01,2e-5,-1e-5])
    _,jvp=jax.jvp(state_parameter_ad,(jnp.array(x),),(jnp.array(d),))
    check(label+' mixed state/coupling JVP',np.linalg.norm(np.asarray(jvp)-J@d),0.,2e-12,max(np.linalg.norm(J@d),1.))
    h=.005;yp,_,_=oracle_value_jacobian(x+h*d,return_diagnostics=True);ym,_,_=oracle_value_jacobian(x-h*d,return_diagnostics=True)
    fd=(yp-ym)/(2*h)
    for i,key in enumerate(OUTPUTS):check(label+' mixed FD '+key,(J@d)[i],fd[i],2e-5,floor[i])
    reports.append(dict(x=x.tolist(),outputs=y.tolist(),jacobian=J.tolist(),diagnostics=diag,parameter_ladders=ladders))

x=jnp.array(pack_state(.16,10.,.1));y,J=oracle_value_jacobian(np.asarray(x))
weights=jnp.array([.01]*8+[1.,1.,1.,10.])
grad=jax.jit(jax.grad(lambda z:jnp.dot(weights,state_parameter_ad(z))))(x)
check('JIT reverse weighted functional',np.linalg.norm(np.asarray(grad)-np.asarray(weights)@J),0.,2e-12,max(np.linalg.norm(np.asarray(weights)@J),1.))
try:jax.jacfwd(jax.jacrev(state_parameter_ad))(x)
except (ValueError,NotImplementedError) as exc:second=dict(supported=False,error_type=type(exc).__name__,message=str(exc).split('\n')[0])
else:second=dict(supported=True,error_type=None,message='Unexpected success')
check('higher derivatives explicitly unsupported',float(not second['supported']),1.,1e-12)
bad=np.asarray(x).copy();bad[3]=0.
check('invalid staged coupling returns NaN',float(np.all(np.isnan(np.asarray(jax.jit(state_parameter_ad)(jnp.array(bad)))))),1.,1e-12)
paths=['rmf_reference/gm1_finite_t.py','rmf_phase/stability.py','rmf_ad/gm1_implicit_ad.py']
report=dict(passed=all(c['passed'] for c in checks),checks=checks,states=reports,input_order=INPUTS,output_order=OUTPUTS,second_derivative_probe=second,runtime_seconds=time.monotonic()-start,
            frozen_sha256={path:hashlib.sha256((ROOT.parent/path).read_bytes()).hexdigest() for path in paths},
            environment=dict(python=sys.version,executable=sys.executable,jax=jax.__version__,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),devices=[str(d) for d in jax.devices()]))
(ROOT/'validation_results.json').write_text(json.dumps(report,indent=2)+'\n')
print(f"{sum(c['passed'] for c in checks)}/{len(checks)} checks pass in {report['runtime_seconds']:.2f}s")
for c in checks:
    if not c['passed']:print('FAIL',c)
raise SystemExit(0 if report['passed'] else 1)
