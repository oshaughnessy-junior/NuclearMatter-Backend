from pathlib import Path
import json,sys,importlib.util,numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('balance_grid_current_audit',P/'grid.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
checks=[dict(name='same GF EC/PC event current',passed=m.EC.E.GF==m.PC.GF),dict(name='same Vud EC/PC event current',passed=m.EC.E.VUD==m.PC.VUD),dict(name='same electron physical mass',passed=m.EC.ME==m.PC.ME==m.BETA.K.ME)]
for T,n in m.STATES:
 b,st=m.material(T,n,.5)
 checks.append(dict(name='actual pair density and charge '+str((T,n)),passed=st['electron']['particle_density']>0 and st['electron']['antiparticle_density']>0 and abs((st['electron']['particle_density']-st['electron']['antiparticle_density'])/(n*.5)-1)<1e-10))
for c in checks:c['passed']=bool(c['passed'])
out=dict(checks=checks,passed=sum(bool(c['passed'])for c in checks),failed=[c for c in checks if not c['passed']]);(P/'current_contract_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
