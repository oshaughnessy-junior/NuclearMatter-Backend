"""Complete independent forward/refinement/ledger review, keeping coarse fail."""
from pathlib import Path
import importlib.util,sys,json,math,hashlib
import numpy as np
P=Path(__file__).parent;ROOT=P.parent;A=ROOT/'hot_electron_capture_moments'
sp=importlib.util.spec_from_file_location('hotEC_independent_probe_only',P/'probe.py');Q=importlib.util.module_from_spec(sp);sys.modules[sp.name]=Q;sp.loader.exec_module(Q);C=Q.C
x=json.loads((P/'results.json').read_text());checks=list(x['checks']);b,meta=C.matched();fine=Q.integrate(b,32,12,10,70.);(P/'independent_fine.json').write_text(json.dumps(fine,indent=2)+'\n')
def ck(name,er,gate=3e-6):checks.append(dict(name=name,error=float(er),gate=gate,passed=bool(math.isfinite(er)and abs(er)<=gate)))
ck('fine independent proton density',fine['density']/3e-8-1,3e-8)
old=x['rows'][-1];names=list(C.NAMES)
for scenario in['empty','LTE','reverse_heldout']:
 for direction in['F','B']:
  if scenario=='empty'and direction=='B':continue
  for i,k in enumerate(names):ck('independent24to32 '+scenario+direction+k,old['scenarios'][scenario][direction][i]/fine['scenarios'][scenario][direction][i]-1)
# Fresh forward reference on author's unchanged implementation, reverse-affinity
# radiation scenario has never supplied a coefficient or normalization fit.
ref=json.loads((A/'results.json').read_text())['rules'][-1];held=C.integrate(b,proton_order=28,electron_order=48,initial_order=12,outgoing_order=12,phi_order=12,tail=60.,scenarios=[dict(name='reverse_heldout',mode='FD',mu_nu=b.equilibrium_mu_nue+.23*b.T)])
for scenario in['empty','LTE','reverse_heldout']:
 r=(held if scenario=='reverse_heldout'else ref)['scenarios'][scenario];z=fine['scenarios'][scenario]
 for di,direction in[('F','forward'),('B','backward')]:
  if scenario=='empty'and di=='B':continue
  for i,k in enumerate(names):ck('independent fine vs source '+scenario+di+k,z[di][i]/r[direction][k]['value']-1)
 gross=z['F'][0]+z['B'][0];scale=z['F'][1]+z['B'][1];net=z['net'];ck('direct escaped material energy '+scenario,(net[1]-net[5])/scale,3e-12);ck('restinclusive material energy '+scenario,(net[4]-net[3]-net[2]+net[1])/scale,3e-10)
 if scenario=='LTE':
  ck('LTE number residual independent',net[0]/gross,3e-12);ck('LTE energy residual independent',net[1]/scale,3e-12)
 if scenario=='reverse_heldout':
  ck('inverse dominates independent new FD affinity',0 if net[0]<0 and net[1]<0 else 1,0);ck('inverse affinity ratio independently occupied',z['F'][0]/z['B'][0]/math.exp(-.23)-1,3e-12);ck('entropy nonnegative independent',0 if z['entropy']>0 else 1,0);ck('entropy equals actual affinity production',z['entropy']/(-.23*net[0])-1,3e-12)
ck('fresh metadata matches authored EOS state',0 if meta==json.loads((A/'MATCHED_STATE.json').read_text())else 1,0);ck('fresh metadata matches PC EOS state',0 if meta==json.loads((ROOT/'positron_capture_moments/MATCHED_STATE.json').read_text())else 1,0)
for p,sha in C.PINS.items():ck('frozen upstream '+p,0 if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha else 1,0)
ck('author files unchanged',0 if all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in x['before'].items())else 1,0)
new={k:v for k,v in x.items()if k not in['checks']};new.update(checks=checks,independent32=fine,new_reverse_reference=held,passed=sum(v['passed']for v in checks),failed=[v for v in checks if not v['passed']],qualified_scope='selected exact common bath EC/inverse moments; retained independent coarse density failure, no combined inventory/opacity');(P/'verification.json').write_text(json.dumps(new,indent=2)+'\n');print(json.dumps(dict(passed=new['passed'],failed=new['failed'],independent_fine_seconds=fine['seconds']),indent=2))
