"""Separate endpoint cutoff and beta phase-coordinate crosschecks."""
import json,time
from pathlib import Path
import element as E
checks=[]
def check(name,error,limit):
 checks.append(dict(name=name,error=float(error),limit=limit,passed=bool(abs(error)<=limit)))
def dump(name,obj):
 import numpy as np
 (Path(__file__).parent/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item()if isinstance(x,np.generic)else str(x)))
start=time.monotonic();old=json.loads((Path(__file__).parent/'results.json').read_text());out=[]
for tag,s in [('initial',old['runs'][0]['initial']),('endpoint',old['runs'][-1]['final'])]:
 fine=next(x['fine']for x in old['source_refinement']if x['tag']==('initial'if tag=='initial'else'changed_endpoint'))
 a=E.PC.integral(s['bath'],order_n=32,order_e=52,order_angle=12,tail=60.,budget=90.)
 for key,idx in [('R',0),('Q',1)]:check(tag+' isolated PC tail60/70 '+key,abs(a['net'][0][idx]/fine['pc_'+key]-1),1e-8)
 b=E.H.integrate(E.K.Bath(**s['bath']),rest_order=24,parent_order=20,orient_order=10,phi_order=12,coordinate='electron',scenarios=[dict(name='empty',mode='empty')],budget_seconds=90.)
 for key,k in [('R','number'),('Q','neutrino_energy')]:check(tag+' independent beta rest phase coordinate '+key,abs(b['scenarios']['empty']['net'][k]/fine['beta_'+key]-1),1e-7)
 out.append(dict(tag=tag,pc_tail60=a,beta_electron_coordinate=b))
# Exact-state cache does not load any saved physical sources.
cache=E.Sources();cache.cache[(2.0.hex(),.3.hex(),E.NB.hex(),(28,44,10,60.),(16,12,6,8))]={'test_exact_key':True}
check('exact state key reused',float(cache.get(E.state(2.,.3))!={'test_exact_key':True}),0.)
check('current vacuum current source pinned',float('muon_vacuum/vacuum_cc.py'not in cache.pin['sources']),0.)
dump('extra_controls.json',dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],cases=out,identity=E.identity(),seconds=time.monotonic()-start));print(json.dumps(dict(checks=checks,seconds=time.monotonic()-start),indent=2))
