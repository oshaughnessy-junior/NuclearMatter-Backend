"""Independent raw-RMF-coupling Jacobian review at a new warm state."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import jax
import jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rmf_parameter_ad'))
import gm1_parameter_ad as a
checks=[]
def check(name,actual,expected,tol=3e-6,scale=1e-7):
    error=float(np.linalg.norm(np.asarray(actual)-expected)/max(np.linalg.norm(np.asarray(expected)),scale))
    checks.append(dict(name=name,actual=np.asarray(actual).tolist(),expected=np.asarray(expected).tolist(),error=error,passed=error<tol))

x=a.pack_state(.27,17.,.22,{'C_sigma':a.GM1['C_sigma']*1.005,'C_omega':a.GM1['C_omega']*.997,'C_rho':a.GM1['C_rho']*1.01,'b':a.GM1['b']*1.002,'c':a.GM1['c']*.998})
y,J,diag=a.oracle_value_jacobian(x,return_diagnostics=True)
checks.append(dict(name='primal residual before derivative interpretation',actual=diag['max_equation_residual'],passed=diag['max_equation_residual']<2e-9))
checks.append(dict(name='positive full species Hessian',actual=diag['chemical_hessian_eigenvalues_mev_fm3'],passed=min(diag['chemical_hessian_eigenvalues_mev_fm3'])>0))
def primal(z):
    coordinates,p=a.unpack(z);s=a.state(*coordinates,params=p)
    return np.array([getattr(s,k) for k in a.OUTPUTS]),s.max_normalized_equation_residual
check('callback exact primal identity',a.state_parameter_ad(jnp.array(x)),primal(x)[0],1e-12)
check('JIT jacfwd all inputs',jax.jit(jax.jacfwd(a.state_parameter_ad))(jnp.array(x)),J,1e-12)
check('JIT jacrev all inputs',jax.jit(jax.jacrev(a.state_parameter_ad))(jnp.array(x)),J,1e-12)

# Independently recover the species chemical Hessian from density perturbations,
# with the other species held fixed and all supplied raw couplings unchanged.
pair=x[0]*np.array([1-x[2],x[2]]);Hfd=np.zeros((2,2))
for j in range(2):
    h=1e-4*pair[j];v=np.zeros(2);v[j]=h;mus=[]
    for pp in [pair-v,pair+v]:
        z=x.copy();z[0]=sum(pp);z[2]=pp[1]/sum(pp)
        yp,_=primal(z);mus.append(yp[[6,7]])
    Hfd[:,j]=(mus[1]-mus[0])/(2*h)
check('independent full species Hessian eigenvalues',np.linalg.eigvalsh((Hfd+Hfd.T)/2),diag['chemical_hessian_eigenvalues_mev_fm3'],2e-6)
checks.append(dict(name='independent species-Hessian stability',actual=np.linalg.eigvalsh((Hfd+Hfd.T)/2).tolist(),passed=min(np.linalg.eigvalsh((Hfd+Hfd.T)/2))>0))

steps=[.002,.001,.0005];ladders=[];maxres=0.
for j in range(3,8):
    estimates=[]
    for frac in steps:
        h=frac*abs(x[j]);v=np.zeros(8);v[j]=h
        yp,rp=primal(x+v);ym,rm=primal(x-v);maxres=max(maxres,rp,rm)
        estimates.append((yp-ym)/(2*h))
    rich=(4*estimates[-1]-estimates[-2])/3
    for k,name in enumerate(a.OUTPUTS):
        check(a.INPUTS[j]+' independent derivative '+name,J[k,j],rich[k],2e-6,1e-5)
    check(a.INPUTS[j]+' Richardson ladder convergence',(4*estimates[-2]-estimates[0])/3,rich,2e-6)
    ladders.append(dict(parameter=a.INPUTS[j],fractional_steps=steps,finite_differences=np.asarray(estimates).tolist(),richardson=rich.tolist()))
checks.append(dict(name='all perturbed primal residual gates',actual=maxres,passed=maxres<2e-9))

I={k:i for i,k in enumerate(a.OUTPUTS)}
# Explicit reduced vector energies independently fix the Gibbs-Duhem work term.
Co_exact=a.HBARC*x[0]**2/2
Cr_exact=a.HBARC*(x[0]*(2*x[2]-1))**2/8
for j,expected in [(4,Co_exact),(5,Cr_exact)]:
    for name in ['energy_density','free_energy_density','pressure']:
        check('explicit vector derivative '+name+' '+a.INPUTS[j],J[I[name],j],expected,1e-12)
    check('vector entropy derivative '+a.INPUTS[j],J[I['entropy_density'],j],0.,1e-12)

direction=np.array([.01,1.,-.02,.1,-.08,.04,2e-5,-1e-5])
cot=np.linspace(-.2,.3,12)
_,forward=jax.jvp(a.state_parameter_ad,(jnp.array(x),),(jnp.array(direction),))
_,pullback=jax.vjp(a.state_parameter_ad,jnp.array(x))
check('mixed state/coupling firstorder JVP',forward,J@direction,1e-12)
check('mixed JVP VJP dot-product',np.dot(cot,np.asarray(forward)),np.dot(np.asarray(pullback(jnp.array(cot))[0]),direction),1e-12)
try:jax.jacfwd(jax.jacrev(a.state_parameter_ad))(jnp.array(x))
except (ValueError,NotImplementedError) as e:checks.append(dict(name='nested AD rejects unsupported higher rule',passed=True,error=str(e).split('\n')[0]))
else:checks.append(dict(name='nested AD rejects unsupported higher rule',passed=False))

out=dict(passed=all(c['passed'] for c in checks),checks=checks,input_order=a.INPUTS,x=x.tolist(),primal=y.tolist(),jacobian=J.tolist(),diagnostics=diag,ladders=ladders,
         source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(a.__file__),ROOT/'rmf_reference/gm1_finite_t.py',ROOT/'rmf_phase/stability.py',ROOT/'rmf_ad/gm1_implicit_ad.py']},
         environment=dict(executable=sys.executable,jax=jax.__version__,numpy=np.__version__))
Path(__file__).with_name('parameter_ad_review.json').write_text(json.dumps(out,indent=2,default=lambda z:z.item())+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[c for c in checks if not c['passed']]),indent=2,default=lambda z:z.item()))
