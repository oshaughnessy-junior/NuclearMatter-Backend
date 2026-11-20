import importlib.util,sys,json,math,hashlib
from pathlib import Path
from scipy.special import kve
from nmbackend.constants import M_E,HBARC
from nmbackend.reference.finite_t_leptons import state_mu
P=Path(__file__).resolve().parents[1]
old=P/'pre_quadrature_repair/staged/src/nmbackend/reference/finite_t_leptons.py'
name='nmbackend.reference._initial_finite_t_leptons'
spec=importlib.util.spec_from_file_location(name,old);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
rows=[]
for z in [40.,100.,300.,600.]:
 T=M_E/z
 exact=math.exp(2*math.log(M_E)+math.log(T)+math.log(kve(2,z))-z-math.log(math.pi**2*HBARC**3))
 a=mod.state_mu(T,0.,M_E);b=state_mu(T,0.,M_E)
 rows.append(dict(z=z,expected_each=exact,initial_each=a.particle_density,repaired_each=b.particle_density,
                  initial_relative_error=abs(a.particle_density/exact-1),repaired_relative_error=abs(b.particle_density/exact-1)))
 assert abs(a.particle_density/exact-1)>1e-4
 assert abs(b.particle_density/exact-1)<3e-10
out=dict(initial_source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
 repaired_source_sha256=hashlib.sha256((P/'staged/src/nmbackend/reference/finite_t_leptons.py').read_bytes()).hexdigest(),
 initial_four_failures_reproduced=True,rows=rows)
(P/'evidence/repair_initial_four_reproduction.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
