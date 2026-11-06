"""Bounded unsupervised polytropic PINO development; no RNS labels in loss.

Independent implementation of Ray's physics-residual procedure. This pilot
is not a general nuclear-EOS trained release or a reproduction of its speedup.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from nmbackend.rotating.pino import network
from nmbackend.rotating.pino.keh import make_grid,fields_from_evaluator,residual_and_observables
from nmbackend.rotating.pino.polytrope import branch_input,eos_at_enthalpy


def configuration(seed,number):
    rng=np.random.default_rng(seed)
    return np.column_stack([rng.uniform(.12,.30,number),rng.uniform(.8,.97,number),rng.uniform(.85,1.15,number)])


def single_loss(params,z,index,grid):
    f=fields_from_evaluator(lambda point:network.evaluate(params,z,point),z[-2],z[-1],grid)
    return residual_and_observables(f,grid,lambda h:eos_at_enthalpy(h,index))


def save_params(path,params):
    leaves,_=jax.tree_util.tree_flatten(params)
    np.savez(path,**{f'p{i}':np.asarray(p) for i,p in enumerate(leaves)})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    ap.add_argument('--steps',type=int,default=2000)
    ap.add_argument('--nr',type=int,default=24)
    ap.add_argument('--nmu',type=int,default=8)
    ap.add_argument('--harmonics',type=int,default=3)
    ap.add_argument('--width',type=int,default=16)
    ap.add_argument('--modes',type=int,default=8)
    ap.add_argument('--hidden-layers',type=int,default=2)
    ap.add_argument('--seed',type=int,default=731)
    ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--train-cases',type=int,default=24)
    ap.add_argument('--learning-rate',type=float,default=5e-4)
    ap.add_argument('--wall-seconds',type=int,default=600)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--lbfgs-steps',type=int,default=0)
    ap.add_argument('--lbfgs-radius',type=float,default=.1)
    ap.add_argument('--coverage-grid',action='store_true')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    grid=make_grid(args.nr,args.nmu,args.harmonics)
    params=network.initialize(jax.random.PRNGKey(args.seed),width=args.width,modes=args.modes,hidden_layers=args.hidden_layers)
    if args.resume:
        from evaluate_pino import load_checkpoint
        restored,settings=load_checkpoint(args.resume)
        if any(settings[k]!=getattr(args,k) for k in ('width','modes','hidden_layers')):
            raise ValueError('Resume architecture does not match requested architecture')
        params=restored
    train=configuration(args.seed,args.train_cases)
    if args.coverage_grid:
        import itertools
        coverage=np.array(list(itertools.product((.12,.21,.30),(.8,.885,.97),(.85,1.,1.15))))
        train=np.vstack([train,coverage])
    val=configuration(args.seed+1,8)
    make=lambda data:jnp.stack([branch_input(jnp.asarray(ec),jnp.asarray(q),index=index) for ec,q,index in data])
    train_z,val_z=make(train),make(val)
    def batch_loss(params,z,indices):
        losses,aux=jax.vmap(lambda zz,ii:single_loss(params,zz,ii,grid))(z,indices)
        return jnp.mean(losses),jax.tree_util.tree_map(lambda x:jnp.mean(x,axis=0),aux)
    evaluate=jax.jit(batch_loss)
    grad=jax.jit(jax.value_and_grad(batch_loss,has_aux=True))
    m=jax.tree_util.tree_map(jnp.zeros_like,params)
    v=jax.tree_util.tree_map(jnp.zeros_like,params)
    @jax.jit
    def update(params,m,v,g,step,rate):
        norm=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(g)))
        g=jax.tree_util.tree_map(lambda x:x/jnp.maximum(norm,1.),g)
        m=jax.tree_util.tree_map(lambda a,b:.9*a+.1*b,m,g)
        v=jax.tree_util.tree_map(lambda a,b:.999*a+.001*b*b,v,g)
        out=jax.tree_util.tree_map(lambda p,mm,vv:p-rate*(mm/(1-.9**step))/(jnp.sqrt(vv/(1-.999**step))+1e-8),params,m,v)
        return out,m,v,norm
    log=[]
    best=np.inf
    rng=np.random.default_rng(args.seed+2)
    manifest=vars(args).copy()
    manifest['output']=str(args.output)
    manifest['resume']=str(args.resume) if args.resume else None
    manifest['parameter_prior']={'energy_center':[.12,.30],'axis_ratio':[.8,.97],'index':[.85,1.15]}
    manifest.update(host=socket.gethostname(),devices=[str(d) for d in jax.devices()],jax=jax.__version__,dtype='float64',training_cases=train.tolist(),validation_cases=val.tolist(),supervised_labels=False)
    manifest['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_root=Path(__file__).resolve().parents[2]/'src/nmbackend/rotating/pino'
    manifest['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.glob('*.py')}
    import shutil
    (args.output/'source').mkdir()
    for p in [*source_root.glob('*.py'),Path(__file__)]:
        shutil.copy2(p,args.output/'source'/p.name)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    initial,_=evaluate(params,val_z,jnp.asarray(val[:,2]))
    best=float(initial)
    save_params(args.output/'best.npz',params)
    print('INITIAL_VALIDATION',float(initial),flush=True)
    for step in range(1,args.steps+1):
        sample=rng.choice(len(train),args.batch,replace=False)
        (loss,aux),g=grad(params,train_z[sample],jnp.asarray(train[sample,2]))
        # Cosine restarts every 500 updates with decay, bounded away from zero.
        cycle=(step-1)//500
        phase=((step-1)%500)/500
        rate=args.learning_rate*(.97**cycle)*(.1+.9*(1+np.cos(np.pi*phase))/2)
        params,m,v,norm=update(params,m,v,g,jnp.asarray(step),rate)
        if step==1 or step%50==0:
            validation,diag=evaluate(params,val_z,jnp.asarray(val[:,2]))
            record={'step':step,'seconds':time.monotonic()-started,'train_loss':float(loss),'validation_loss':float(validation),'gradient_norm':float(norm),'validation':{k:np.asarray(x).tolist() for k,x in diag.items()}}
            log.append(record)
            print(json.dumps(record),flush=True)
            (args.output/'progress.json').write_text(json.dumps(log,indent=2)+'\n')
            if not np.isfinite(float(validation)):
                save_params(args.output/'failed.npz',params)
                raise RuntimeError('Nonfinite physics loss; no sample exclusion or fallback')
            if float(validation)<best:
                best=float(validation)
                save_params(args.output/'best.npz',params)
            if time.monotonic()-started>args.wall_seconds:
                break
    if args.lbfgs_steps and time.monotonic()-started<args.wall_seconds:
        from scipy.optimize import minimize
        from jax.flatten_util import ravel_pytree
        flat,unravel=ravel_pytree(params)
        full_indices=jnp.asarray(train[:,2])
        origin=flat
        parameter_scale=.01
        objective=jax.jit(jax.value_and_grad(lambda x:batch_loss(unravel(origin+parameter_scale*x),train_z,full_indices)[0]))
        iterations=[0]
        def fun(x):
            loss,g=objective(jnp.asarray(x))
            if not np.isfinite(float(loss)) or not np.all(np.isfinite(g)):
                raise RuntimeError('Nonfinite full-batch objective/gradient')
            return float(loss),np.asarray(g)
        def callback(x):
            nonlocal best
            iterations[0]+=1
            if iterations[0]==1 or iterations[0]%10==0:
                candidate=unravel(origin+parameter_scale*jnp.asarray(x))
                validation,diag=evaluate(candidate,val_z,jnp.asarray(val[:,2]))
                record={'phase':'lbfgs','step':iterations[0],'seconds':time.monotonic()-started,'validation_loss':float(validation),'validation':{k:np.asarray(a).tolist() for k,a in diag.items()}}
                log.append(record)
                print(json.dumps(record),flush=True)
                (args.output/'progress.json').write_text(json.dumps(log,indent=2)+'\n')
                if float(validation)<best:
                    best=float(validation)
                    save_params(args.output/'best.npz',candidate)
                if time.monotonic()-started>args.wall_seconds:
                    raise StopIteration
        bound=args.lbfgs_radius/parameter_scale
        result=minimize(fun,np.zeros_like(np.asarray(flat)),jac=True,method='L-BFGS-B',bounds=[(-bound,bound)]*len(flat),callback=callback,options={'maxiter':args.lbfgs_steps,'maxls':30,'maxcor':20,'ftol':1e-12,'gtol':1e-8})
        params=unravel(origin+parameter_scale*jnp.asarray(result.x))
        (args.output/'lbfgs.json').write_text(json.dumps({'success':bool(result.success),'message':str(result.message),'iterations':int(result.nit),'function_calls':int(result.nfev),'final_objective':float(result.fun)},indent=2)+'\n')
    save_params(args.output/'last.npz',params)
    summary={'initial_validation_loss':float(initial),'best_validation_loss':best,'steps':step,'seconds':time.monotonic()-started,'scope':'polytropic physics-only pilot; no RNS forward/inference/gradient qualification implied'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
