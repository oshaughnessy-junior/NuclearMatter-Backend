"""Execute only actual pinned NMB cold pure functions in existing JAX64 runtime."""
from pathlib import Path
import sys,math,ast,json,time,hashlib
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(P));import cold_beta as own
source=ROOT/'rmf_reference/source_snapshot';constants={};exec((source/'constants.py').read_text(),constants)
ns=dict(jax=jax,jnp=jnp,math=math,_PI2=math.pi**2,HBARC=constants['HBARC'],M_E=constants['M_E'],M_MU=constants['M_MU'])
for path,names in [(P/'source_snapshot/leptons.py',{'eps_kin_one','_kf_of_mu','lepton_sector'}),(source/'rmft.py',{'_kf','_rho_s_one','_eps_kin_one','_solve_phi','_leptons','_vector_fields','_solve_xp','_eps_p_from_state'})]:
 tree=ast.parse(path.read_text());selected=[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name in names]
 if path.name=='rmft.py':selected+=[x for x in tree.body if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='GM1'for t in x.targets)]
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
@jax.jit
def actual(n):
 x=ns['_solve_xp'](n,ns['GM1']);mu,ne,nmu,energy=ns['_leptons'](x*n);phi=ns['_solve_phi']((1-x)*n,x*n,ns['GM1']);eps,pres,mun=ns['_eps_p_from_state'](n,x,phi,ns['GM1']);return x,mu*own.H,ne/n,nmu/n,phi*own.H,eps,pres,mun
checks=[];rows=[];start=time.monotonic()
for density in (.12,.2,.27808452804707523,.34397893887478437,.6):
 x,mu,ye,ym,phi,eps,pres,mun=[float(x)for x in actual(jnp.array(density))];m=own.state(density,electron_mass=constants['M_E']);default=own.state(density)
 for name,a,b in [('Yp',x,m.Yp),('mu',mu,m.mu_lepton),('Ye',ye,m.Ye),('Ymu',ym,m.Ymu),('Phi',phi,m.nuclear.phi_mev),('total_energy',eps,m.energy_density),('total_pressure',pres,m.pressure),('total_mun',mun,m.nuclear.mu_n_mev)]:
  error=abs(a-b)/max(abs(b),1e-10);checks.append(dict(name=str(density)+' pinnedactual '+name,error=error,tolerance=3e-10,passed=bool(error<3e-10)))
 rows.append(dict(nB=density,pinned_source=dict(Yp=x,mu=mu,Ye=ye,Ymu=ym,Phi=phi,energy=eps,pressure=pres,mu_n=mun),own_matched_legacy_electron_mass=m.asdict(),default_minus_legacy=dict(Yp=default.Yp-m.Yp,mu=default.mu_lepton-m.mu_lepton)))
(P/'pinned_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,runtime_seconds=time.monotonic()-start,scope='Actual pinned selectedcoldfunctions executed with JAX64, not fullNMBpackage/test suite nor externalphysicalvalidation',sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in (source/'rmft.py',source/'constants.py',P/'source_snapshot/leptons.py')},mass_convention=dict(own_default_electron=own.ME,pinned_electron=constants['M_E'],muon=own.MMU,common_bare_nucleon=939.)),indent=2)+'\n');print(sum(c['passed']for c in checks),'/',len(checks),'runtime',time.monotonic()-start)
for c in checks:
 if not c['passed']:print('FAIL',c)
