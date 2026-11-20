"""Independent first-order callback/implicit-AD review at new warm states."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import jax
import jax.numpy as jnp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rmf_ad'))
import gm1_implicit_ad as a
from gm1_finite_t import state,gas
from stability import implicit_hessian

checks=[];samples=[]
def check(name,actual,expected,tol=1e-6,scale=1e-12):
    actual=np.asarray(actual);expected=np.asarray(expected)
    error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),scale))
    checks.append(dict(name=name,actual=actual.tolist(),expected=expected.tolist(),relative_error=error,passed=error<tol))
I={name:i for i,name in enumerate(a.OUTPUTS)}
def direct(x):
    s=state(*map(float,x));return np.array([getattr(s,k) for k in a.OUTPUTS])

for x in [np.array([.2,15.,.35]),np.array([.4,40.,.2])]:
    tag=str(x.tolist());y,J,diag=a.oracle_value_jacobian(x,return_diagnostics=True)
    ss=state(*x);H,_=implicit_hessian(ss)
    check(tag+' stable local chemical Hessian',float(np.linalg.eigvalsh(H)[0]>0),1.,1e-12)
    check(tag+' exact primal all12',a.state_ad(jnp.array(x)),direct(x),1e-12)
    check(tag+' JIT jacfwd',jax.jit(jax.jacfwd(a.state_ad))(jnp.array(x)),J,1e-12)
    check(tag+' JIT jacrev',jax.jit(jax.jacrev(a.state_ad))(jnp.array(x)),J,1e-12)
    # Independent coordinate finite differences of direct converged reference.
    jacobians=[]
    for factor in [1.,.5]:
        step=factor*np.array([2e-4,2e-2,2e-4]);JJ=np.zeros((12,3))
        for j in range(3):
            v=np.zeros(3);v[j]=step[j]
            JJ[:,j]=(direct(x+v)-direct(x-v))/(2*step[j])
        jacobians.append(JJ)
    rich=(4*jacobians[1]-jacobians[0])/3
    for k,name in enumerate(a.OUTPUTS):
        check(tag+' Richardson direct Jacobian '+name,J[k],rich[k],2e-6,1e-5)
    # Entropy Maxwell identities use independent state chemical-temperature FDs.
    h=.02
    sm=state(x[0],x[1]-h,x[2]);sp=state(x[0],x[1]+h,x[2])
    muT=(np.array([sp.mu_n_mev,sp.mu_p_mev])-np.array([sm.mu_n_mev,sm.mu_p_mev]))/(2*h)
    check(tag+' entropy_n=-mu_bar_T',J[I['entropy_density'],0],-np.array([1-x[2],x[2]])@muT,2e-6)
    check(tag+' entropy_Y=-n mu_diff_T',J[I['entropy_density'],2],-x[0]*(muT[1]-muT[0]),2e-6)
    check(tag+' entropy positive frozen Cv',float(J[I['entropy_density'],1]>0),1.,1e-12)
    direction=np.array([.015,1.3,-.02]);cotangent=np.linspace(-.7,.8,12)
    _,jvp=jax.jvp(a.state_ad,(jnp.array(x),),(jnp.array(direction),))
    _,pullback=jax.vjp(a.state_ad,jnp.array(x));vjp=pullback(jnp.array(cotangent))[0]
    check(tag+' JVP/VJP dot-product',np.dot(cotangent,np.asarray(jvp)),np.dot(np.asarray(vjp),direction),1e-12)
    # A nonlinear downstream function exercises differentiating ordinary JAX
    # algebra around the callback without requesting unsupported second order.
    def loss(z):
        yy=a.state_ad(z)
        return jnp.log(yy[I['pressure']])+yy[I['entropy_density']]/z[0]+.0001*(yy[I['mu_n_mev']]-yy[I['mu_p_mev']])**2
    def lossdirect(z):
        yy=direct(z)
        return np.log(yy[I['pressure']])+yy[I['entropy_density']]/z[0]+.0001*(yy[I['mu_n_mev']]-yy[I['mu_p_mev']])**2
    gd=np.zeros(3)
    for j,h in enumerate([1e-5,1e-3,1e-5]):
        v=np.zeros(3);v[j]=h;gd[j]=(lossdirect(x+v)-lossdirect(x-v))/(2*h)
    check(tag+' nonlinear downstream JIT grad',jax.jit(jax.grad(loss))(jnp.array(x)),gd,2e-6)
    samples.append(dict(x=x.tolist(),primal=y.tolist(),jacobian=J.tolist(),independent_richardson=rich.tolist(),diagnostics=diag))

# Pair-sensitive temperature moments; add a mu=0 case absent in AD author tests.
for T,nu,m in [(100./a.HBARC,0.,939./a.HBARC),(100./a.HBARC,.5,300./a.HBARC)]:
    response=a.thermal_response(T,nu,m);h=.0002*T
    fm2=gas(T-2*h,nu,m);fm=gas(T-h,nu,m);fp=gas(T+h,nu,m);fp2=gas(T+2*h,nu,m)
    for key,field in [('n_T','n'),('rho_s_T','scalar'),('s_T','entropy')]:
        fd=(fm2[field]-8*fm[field]+8*fp[field]-fp2[field])/(12*h)
        check(f'pair-sensitive moment T{T} nu{nu} {key}',response[key],fd,1e-7,1e-10)

for bad in [np.array([.2,-1.,.3]),np.array([.2,15.,1.]),np.array([np.nan,15.,.3])]:
    rejected=False
    try:a.oracle_value_jacobian(bad)
    except (ArithmeticError,ValueError):rejected=True
    checks.append(dict(name='strict invalid preflight '+str(bad),passed=rejected))
    checks.append(dict(name='JIT invalid NaN sentinel '+str(bad),passed=bool(np.all(np.isnan(np.asarray(jax.jit(a.state_ad)(jnp.array(bad))))))))

try:jax.jacrev(jax.jacfwd(a.state_ad))(jnp.array([.2,15.,.35]))
except (ValueError,NotImplementedError) as e:
    checks.append(dict(name='reverse-over-forward second AD rejected',passed=True,error=str(e).split('\n')[0]))
else:checks.append(dict(name='reverse-over-forward second AD rejected',passed=False))

out=dict(passed=all(c['passed'] for c in checks),checks=checks,samples=samples,
         source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(a.__file__),ROOT/'rmf_reference/gm1_finite_t.py',ROOT/'rmf_phase/stability.py']},
         environment=dict(python=sys.version,executable=sys.executable,jax=jax.__version__,numpy=np.__version__))
Path(__file__).with_name('rmf_ad_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[x for x in checks if not x['passed']]),indent=2))
