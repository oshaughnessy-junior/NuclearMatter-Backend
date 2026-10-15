"""Bounded forward-only comparison; no upstream driver imports or writes."""
from __future__ import annotations
from pathlib import Path
import hashlib, importlib.util, json, math, platform, shutil, subprocess, time
import numpy as np
from momentum_reduction import solve,HC

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load_reference():
    spec=importlib.util.spec_from_file_location('external_gm1_forward_internal_reference',ROOT/'rmf_reference/gm1_finite_t.py')
    import sys
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    started=time.monotonic(); checks=[]; rows=[]
    def check(name,error,tol):
        checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<=tol)))
    ref=load_reference()
    for state in [(.16,10.,.1),(.24,20.,.3)]:
        # These fresh forward solves are internal values, never external oracle.
        internal=ref.state(*state).asdict()
        ladder=[solve(*state,order=n,tail=tail) for n,tail in [(96,40.),(160,40.),(256,40.),(256,60.)]]
        finest=ladder[-1]
        for name in ['phi_mev','omega_mev','rho_mev','dirac_mass_mev','mu_n_mev','mu_p_mev','energy_density','pressure','entropy_density','free_energy_density','scalar_density','antiparticle_density']:
            scale=max(abs(internal[name]),1e-100)
            check(f'{state}:internal-forward:{name}',abs(finest[name]-internal[name])/scale,2e-9)
            for a,b in zip(ladder[:-1],ladder[1:]):
                check(f'{state}:quadrature-{a["order"]}-{a["tail"]}->{b["order"]}-{b["tail"]}:{name}',abs(a[name]-b[name])/max(abs(b[name]),1e-100),2e-9)
        for name in ['normalized_scalar_residual','normalized_n_residual','normalized_p_residual']:
            check(f'{state}:root:{name}',abs(finest[name]),2e-10)
        identity=finest['energy_density']+finest['pressure']-state[1]*finest['entropy_density']-finest['mu_n_mev']*finest['nn']-finest['mu_p_mev']*finest['np']
        check(f'{state}:Euler',abs(identity)/finest['energy_density'],2e-10)
        rows.append(dict(state=state,internal=internal,momentum_ladder=ladder))
    # Existing compiler only. A missing header is recorded, no build/install.
    compiled=subprocess.run(['clang++','-std=c++11','-fsyntax-only',str(HERE/'o2scl_fixture.cpp')],text=True,capture_output=True,timeout=10)
    prefixes=[Path(x) for x in ['/usr/local/lib','/opt/homebrew/lib','/Users/rossma/miniconda3/lib','/Users/rossma/miniconda3/envs/junior_tools/lib']]
    avail=dict(platform=platform.platform(),python=platform.python_version(),compiler=shutil.which('clang++'),o2scl_config=shutil.which('o2scl-config'),library_candidates=[str(q) for p in prefixes if p.exists() for q in p.glob('*o2scl*')],compile_status=compiled.returncode,compile_stdout=compiled.stdout,compile_stderr=compiled.stderr,old_checkout_has_source=Path('/private/tmp/q2c-o2scl-oracle-source/src/eos/eos_had_rmf.cpp').exists(),external_execution=False)
    sources={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'rmf_reference/gm1_finite_t.py',ROOT/'external_gm1/comparison_contract.json',HERE/'momentum_reduction.py',HERE/'o2scl_fixture.cpp',*sorted((HERE/'sources').glob('*'))]}
    result=dict(status='SAME_EQUATION_REDUCTION_ONLY_EXTERNAL_CODE_GATE_OPEN',o2scl_commit='d68532ff31e59dac65844c3d387e1388cb71a770',runtime_seconds=time.monotonic()-started,availability=avail,source_sha256=sources,rows=rows,checks=checks,passed=sum(x['passed'] for x in checks),failed=sum(not x['passed'] for x in checks))
    (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['status','runtime_seconds','passed','failed']}))
    return int(result['failed']!=0)

if __name__=='__main__':raise SystemExit(main())
