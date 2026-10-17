"""FinishONLYpreregisteredindependentbeta base probes afterfineforwardresolved.

OriginalfullbasepipelinePCforwardFAILandmissingbeta remainunmodified.
"""
from pathlib import Path
from types import SimpleNamespace
import json,sys,importlib.util,time
import numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;s=importlib.util.spec_from_file_location('weak_beta_continuation_unique_oracles',P/'oracles.py');O=importlib.util.module_from_spec(s);sys.modules[s.name]=O;s.loader.exec_module(O);rows=[];checks=[];start=time.monotonic()
def ck(name,e,g):checks.append(dict(name=name,error=float(e),gate=g,passed=bool(abs(e)<=g)))
def enc(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(str(type(x)))
for i,q in enumerate(O.LOCK['baths']):
 fine=json.loads((P/f'pure_{i}_fine.json').read_text());assert fine['stage']=='COMPLETE'and not fine['failed'];b=SimpleNamespace(**{k:q[k]for k in['T','mu_n','mu_p','mu_e']});rad={n:None if r is None else r['mu_antinue']for n,r in q['radiation'].items()};p=O.LOCK['profiles']['base']['NEUTRON_DECAY'];z=O.BETA.integrate(b,0.,order=p[0],radial=p[1],orient=p[2],phi=p[3],budget=105,radiation_mus=rad,parent_tail=p[4]);ck('basebeta'+str(i)+'actualFDdensity',z['density']/fine['forward']['neutron']['number']-1,3e-9);ck('basebeta'+str(i)+'onshell',z['max_fourmomentum_residual']/O.MN,2e-10);rows.append(dict(bath_index=i,channels=dict(NEUTRON_DECAY=z)))
 (P/'beta_base_continuation.json').write_text(json.dumps(dict(rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],preserved_basePC_failures=[json.loads((P/f'pure_{j}_base.json').read_text())['failed']for j in[0,1]],no_saved_rate_feedback=True,seconds=time.monotonic()-start),default=enc,indent=2)+'\n')
print('BETABASE',sum(c['passed']for c in checks),'FAIL',sum(not c['passed']for c in checks),time.monotonic()-start)
