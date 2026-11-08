"""Compare an independently trained checkpoint to held-out RNS exports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import time
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
from nmbackend.rotating.rns import load_background
from nmbackend.rotating.pino import network
from nmbackend.rotating.pino.keh import make_grid,fields_from_evaluator,residual_and_observables,alpha_from_fields
from nmbackend.rotating.pino.polytrope import branch_input,eos_at_enthalpy


def load_checkpoint(root):
    settings=json.loads((root/'manifest.json').read_text())
    seed=network.initialize(jax.random.PRNGKey(settings['seed']),width=settings['width'],modes=settings['modes'],hidden_layers=settings['hidden_layers'])
    leaves,tree=jax.tree_util.tree_flatten(seed)
    arrays=np.load(root/'best.npz')
    restored=[]
    for i,leaf in enumerate(leaves):
        a=arrays[f'p{i}']
        if a.shape!=leaf.shape or not np.all(np.isfinite(a)):
            raise ValueError('Checkpoint shape/value mismatch')
        restored.append(jnp.asarray(a))
    code=Path(__file__).resolve().parents[2]/'src/nmbackend/rotating/pino'
    for name,expected in settings['source_sha256'].items():
        if hashlib.sha256((code/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Checkpoint is bound to different source: '+name)
    return jax.tree_util.tree_unflatten(tree,restored),settings


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('checkpoint',type=Path)
    ap.add_argument('corpus',type=Path)
    ap.add_argument('output',type=Path)
    args=ap.parse_args()
    params,settings=load_checkpoint(args.checkpoint)
    records=[]
    # RNS labels never enter the loss. Reused RNS cases are development
    # benchmarks, not an untouched final test set after architecture iteration.
    for nr,nmu,nh in ((settings['nr'],settings['nmu'],settings['harmonics']),(64,16,6)):
        grid=make_grid(nr,nmu,nh)
        @jax.jit
        def solve(z,index):
            fields=fields_from_evaluator(lambda point:network.evaluate(params,z,point),z[-2],z[-1],grid)
            _,diag=residual_and_observables(fields,grid,lambda h:eos_at_enthalpy(h,index))
            return diag,fields.value,alpha_from_fields(fields,grid)
        for archive in sorted(args.corpus.glob('*.tar.gz')):
            with tarfile.open(archive) as tar,tempfile.TemporaryDirectory() as d:
                path=Path(d)/'star.csv'
                path.write_bytes(tar.extractfile('result/rns_rotating.csv').read())
                star=load_background(path)
            meta=star.metadata
            index=1/(meta['Gamma']-1)
            z=branch_input(jnp.asarray(meta['e_center']),jnp.asarray(meta['r_ratio']),index=index)
            started=time.perf_counter()
            diag,predicted_fields,predicted_alpha=solve(z,index)
            diag=jax.tree_util.tree_map(lambda a:np.asarray(a).tolist(),diag)
            seconds=time.perf_counter()-started
            from check_keh import reference_fields
            reference=reference_fields(star,grid,stride=2)
            field_error=np.sqrt(np.mean((np.asarray(predicted_fields-reference.value))**2,axis=(0,1))/np.mean(np.asarray(reference.value)**2,axis=(0,1)))
            prior=settings.get('parameter_prior',{'energy_center':[.12,.30],'axis_ratio':[.8,.97],'index':[.85,1.15]})
            in_domain=all(prior[k][0]<=v<=prior[k][1] for k,v in [('energy_center',meta['e_center']),('axis_ratio',meta['r_ratio']),('index',index)])
            records.append({'archive':archive.name,'nr':nr,'nmu':nmu,'in_training_prior':in_domain,'reference':meta,'prediction':diag,'field_rms_relative_error':field_error.tolist(),'mass_relative_error':diag['mass']/meta['Mass']-1,'radius_relative_error':diag['radius']/meta['R_e']-1,'omega_relative_error':diag['omega']/meta['Omega']-1,'seconds_including_compile_if_first':seconds})
            print(json.dumps({k:v for k,v in records[-1].items() if k not in ('reference','prediction')}),flush=True)
    fine=[r for r in records if r['nr']==64 and r['in_training_prior']]
    summary={key:max(abs(r[key]) for r in fine) for key in ('mass_relative_error','radius_relative_error','omega_relative_error')}
    summary.update(checkpoint_sha256=hashlib.sha256((args.checkpoint/'best.npz').read_bytes()).hexdigest(),scope='independent polytropic pilot; no speedup or nuclear-EOS inference qualification')
    summary['in_prior_reference_cases']=len(fine)
    args.output.write_text(json.dumps({'summary':summary,'records':records},indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
