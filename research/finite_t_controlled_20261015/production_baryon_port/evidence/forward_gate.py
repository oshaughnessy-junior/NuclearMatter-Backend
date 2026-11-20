"""Actual accepted capture-bath forward quantities gate later responses."""
from pathlib import Path
import sys,json,hashlib,importlib.util,math
from nmbackend.reference.finite_t_baryons import state_mu,state_density
P=Path(__file__).resolve().parents[1];R=P.parent
path=R/'relativistic_free_eos/free_baryons.py';spec=importlib.util.spec_from_file_location('baryon_port_accepted_physical_eos',path);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
checks=[];rows=[]
for i in [0,1,2]:
 raw=json.loads((R/f'transparent_balance_grid/state_{i}.json').read_text());bath=raw['balanced']['bath'];T=bath['T'];n=raw['balanced']['nB'];y=raw['kinetic_Ye']
 for label,m,mu,target in [('neutron',old.MN,bath['mu_n'],n*(1-y)),('proton',old.MP,bath['mu_p'],n*y)]:
  a=old.state_mu(T,mu,m);b=state_mu(T,mu,m);root=state_density(T,target,m)
  for newkey,oldkey in [('particle_density','density'),('kinetic_energy_density','kinetic_energy_density'),('energy_density','energy_density'),('pressure','pressure'),('entropy_density','entropy_density'),('free_energy_density','free_energy_density')]:
   expected=getattr(a,oldkey);value=getattr(b,newkey);error=abs(value/expected-1);checks.append(dict(name=f'bath{i}/{label}/{newkey}',error=error,gate=3e-9,passed=error<3e-9))
  error=abs(root.particle_density/target-1);checks.append(dict(name=f'bath{i}/{label}/densityroot',error=error,gate=1e-9,passed=error<1e-9));checks.append(dict(name=f'bath{i}/{label}/totalmu',error=abs(root.chemical_potential-mu),gate=3e-9,passed=abs(root.chemical_potential-mu)<3e-9))
  rows.append(dict(T=T,label=label,mass=m,input_mu=mu,target_density=target,state=b.asdict(),root=root.asdict(),accepted=a.asdict()))
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,rows=rows,source_sha256=hashlib.sha256((P/'staged/src/nmbackend/reference/finite_t_baryons.py').read_bytes()).hexdigest(),accepted_eos_sha256=hashlib.sha256(path.read_bytes()).hexdigest());(P/'evidence/forward_gate.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed']);assert not out['failed']
