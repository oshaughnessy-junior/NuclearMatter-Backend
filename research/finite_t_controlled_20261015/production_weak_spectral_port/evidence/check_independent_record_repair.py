"""Narrow record repair: consume frozen payloads only, no physical rerun."""
from pathlib import Path
import sys,json,dataclasses,hashlib,ast
R=Path(__file__).resolve().parents[1];P=R/'pre_independent_record_repair'
sys.path.insert(0,str(R/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_spectral_common as S
assert Path(W.__file__).resolve()==(R/'test_mirror/src/nmbackend/reference/weak_reference.py').resolve()
checks=[]
def ck(name,ok):checks.append(dict(name=name,passed=bool(ok)))
def tuples(v):return tuple(tuples(x)for x in v)if isinstance(v,list)else v

def spectral(d):
 q=dict(d);q.update(bath=W.FreeElectronBath(**d['bath']),controls=W.SpectralControls(**d['controls']),j=W.SpectralQuantity(**d['j']),true_chi=W.SpectralQuantity(**d['true_chi']),geometry=tuples(d['geometry']),differential_coordinates=tuples(d['differential_coordinates']))
 return W.SpectralResult(**q)

def reconstruction(d):
 q=dict(d);rule=d['rule'];e=d['entropy_production']
 q.update(bath=W.FreeElectronBath(**d['bath']),radiation=W.Radiation(**d['radiation']),controls=W.SpectralControls(**d['controls']),rule=W.EnergyRule(tuple(rule['nodes']),tuple(rule['weights']),rule['lower'],rule['upper']),nodes=tuple(spectral(n)for n in d['nodes']),forward=tuple(W.PositiveQuantity(**v)for v in d['forward']),backward=tuple(W.PositiveQuantity(**v)for v in d['backward']),net=tuple(W.SignedQuantity(**v)for v in d['net']),entropy_production=None if e is None else W.SignedQuantity(**e)if'sign'in e else W.PositiveQuantity(**e),units=tuple(d['units']))
 return W.ReconstructionResult(**q)

def identical(obj,d):return json.loads(json.dumps(dataclasses.asdict(obj)))==d

for ch in('EC','PC','NEUTRON_DECAY'):
 for order in(2,4):
  for scenario in('empty','LTE','noneq'):
   f=P/'author/evidence'/f'pilot_{ch}_{order}_{scenario}.json';d=json.loads(f.read_text())['result'];ck('original actual pilot '+f.name,identical(reconstruction(d),d))
review=json.loads((P/'review/GUARD_RESULTS.json').read_text())
for i,d in enumerate(review['pilots']):ck('reviewer valid actual pilot '+str(i),identical(reconstruction(d),d))
for i,w in enumerate(review['witnesses']):
 try:
  (reconstruction if w['accepted_type']=='ReconstructionResult'else spectral)(w['accepted']);rejected=False
 except ValueError:rejected=True
 ck('exact reviewer initial failed witness '+str(i),rejected)
# Every actual original-source incident node remains a byte-identical valid
# record. This is constructor/source identity, NOT a new physical quadrature.
count=0
for ch in('EC','PC'):
 for n in(4,8,16,24,32):
  for first in range(0,n,4):
   f=P/'study/raw'/f'{ch}_{n}_{first:02d}.json';d=json.loads(f.read_text());assert d['complete']
   for row in d['rows']:ck(f'original incident {ch} {n} {row["index"]}',identical(spectral(row['result']),row['result']));count+=1
# Scalar/phase helpers and public dispatcher unchanged byte-for-byte.
for name in('_weak_current.py','_weak_two_body.py','_weak_decay.py','_weak_quadrature.py','_weak_spectral_capture.py','_weak_spectral_decay.py','weak_reference.py'):
 old=P/'author/test_mirror/src/nmbackend/reference'/name;new=R/'test_mirror/src/nmbackend/reference'/name;ck('unchanged physical/public '+name,old.read_bytes()==new.read_bytes())
old=ast.parse((P/'author/staged/src/nmbackend/reference/_weak_spectral_common.py').read_text());new=ast.parse((R/'staged/src/nmbackend/reference/_weak_spectral_common.py').read_text());of={x.name:x for x in old.body if isinstance(x,(ast.ClassDef,ast.FunctionDef))};nf={x.name:x for x in new.body if isinstance(x,(ast.ClassDef,ast.FunctionDef))}
for name in('SpectralQuantity','SpectralControls','EnergyRule','coefficient','empty_coefficient','validate','differential_inputs','dos'):
 ck('unchanged common AST '+name,ast.dump(of[name],include_attributes=False)==ast.dump(nf[name],include_attributes=False))
oloop=next(x for x in of['reconstruct'].body if isinstance(x,ast.For));nloop=next(x for x in nf['_assemble_spectral_nodes'].body if isinstance(x,ast.For))
idx=next(i for i,x in enumerate(oloop.body)if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='base'for t in x.targets))
ck('canonical per-node arithmetic AST exactly moved',[ast.dump(x,include_attributes=False)for x in oloop.body[idx:]]==[ast.dump(x,include_attributes=False)for x in nloop.body])
start=next(i for i,x in enumerate(of['reconstruct'].body)if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='f'for t in x.targets));nstart=next(i for i,x in enumerate(nf['_assemble_spectral_nodes'].body)if isinstance(x,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='f'for t in x.targets))
ck('canonical completed arithmetic AST exactly moved',[ast.dump(x,include_attributes=False)for x in of['reconstruct'].body[start:-1]]==[ast.dump(x,include_attributes=False)for x in nf['_assemble_spectral_nodes'].body[nstart:-1]])
for name in ('SpectralResult','ReconstructionResult'):
 ck('unchanged result fields '+name,[ast.dump(x,include_attributes=False)for x in of[name].body if isinstance(x,ast.AnnAssign)]==[ast.dump(x,include_attributes=False)for x in nf[name].body if isinstance(x,ast.AnnAssign)])
for name in ('flavor','relaxation','absorption_cm_inverse','absorption_km_inverse','kms_log_residual'):
 a=next(x for x in of['SpectralResult'].body if isinstance(x,ast.FunctionDef)and x.name==name);b=next(x for x in nf['SpectralResult'].body if isinstance(x,ast.FunctionDef)and x.name==name)
 ck('unchanged public derived property '+name,ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False))
new_eval_loop=next(x for x in nf['reconstruct'].body if isinstance(x,ast.For))
ck('unchanged physical evaluator loop AST',[ast.dump(x,include_attributes=False)for x in oloop.body[:idx]]==[ast.dump(x,include_attributes=False)for x in new_eval_loop.body])
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],actual_old_pilot_records=21,actual_incident_records=count,source_calls=0,scope='canonical record coherence and exact physics/source identity only; not cryptographic physical provenance')
(R/'evidence/independent_record_repair_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],'PASS',len(out['failed']),'FAIL','incident',count);print(out['failed'])
if out['failed']:sys.exit(1)
