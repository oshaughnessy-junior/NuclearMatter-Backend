from pathlib import Path
import importlib.util,sys,json,math,time
P=Path(__file__).resolve().parent;R=P.parent;sys.path.insert(0,str(P));import matter as M
sp=importlib.util.spec_from_file_location('binding_old_matched_functions',R/'hot_electron_capture_moments/electron_capture.py');EC=importlib.util.module_from_spec(sp);sys.modules[sp.name]=EC;sp.loader.exec_module(EC)
checks=[];rows=[];started=time.monotonic()
def ck(n,a,b,tol):
 err=abs(a-b)/max(abs(a),abs(b),1e-290);checks.append(dict(name=n,a=float(a),b=float(b),relative_error=float(err),gate=tol,passed=bool(err<=tol)))
for T,n,Y in M.LOCK['selected_forward_baths']:
 bath,old=EC.matched(T,n,Y);s=M.state(T,n,Y,photons=False);rows.append(dict(new=s.asdict(),old=old))
 for name,new in [('neutron',s.neutron),('proton',s.proton),('electron',s.electron)]:
  z=old[name]
  for field in ['chemical_potential','energy_density','pressure','entropy_density','free_energy_density','dn_dmu_fixed_temperature','dn_dtemperature_fixed_mu']:
   ck(str((T,n,Y,name,field)),getattr(new,field),z[field],M.LOCK['gates']['forward_fields_relative'])
  for field,ofield in [('particle_density','density'if name!='electron'else'particle_density')]:ck(str((T,name,field)),getattr(new,field),z[ofield],3e-9)
  cvfield='cv_fixed_net_density'if name=='electron'else'cv_fixed_particle_density';oldcv='cv_fixed_net_density'if name=='electron'else'cv_fixed_density';ck(str((T,name,'Cv')),getattr(new,cvfield),z[oldcv],3e-8)
 ck(str((T,n,Y,'electron net')),s.electron.net_density,Y*n,2e-9)
 ck(str((T,n,Y,'pair net reconstructed')),s.electron.particle_density-s.electron.antiparticle_density,Y*n,2e-9)
out=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,states=rows,seconds=time.monotonic()-started,old_source=str(EC.__file__))
(P/'forward_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'])
if out['failed']:raise SystemExit(1)
