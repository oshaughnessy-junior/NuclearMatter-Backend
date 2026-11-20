from pathlib import Path
import sys,importlib.util,json,dataclasses,math,hashlib,ast,time
P=Path(__file__).resolve().parent;A=P.parent/'production_weak_spectral_port';sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'repaired_copy/src'))
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_spectral_common as S, _weak_quadrature as Q
s=importlib.util.spec_from_file_location('own_exact_typed_records',P/'record_replay_helpers.py');H=importlib.util.module_from_spec(s);s.loader.exec_module(H)
checks=[];exceptions=[];start=time.monotonic()
def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)))
def reject(n,fn):
 try:fn()
 except ValueError as e:ck(n,True);exceptions.append(dict(name=n,message=str(e)))
 except Exception as e:ck(n,False);exceptions.append(dict(name=n,unexpected=repr(e)))
 else:ck(n,False)
g=json.loads((P/'GUARD_RESULTS.json').read_text())
for w in g['witnesses']:
 reject('oldwrongacceptance '+w['name'],lambda w=w:(H.reconstruction if w['accepted_type']=='ReconstructionResult'else H.spectrum)(W,w['accepted']))
for i,z in enumerate(g['pilots']):ck('ownactualpilotidentity '+str(i),H.json_identity(H.reconstruction(W,z),z))
for ch in ['EC','PC','NEUTRON_DECAY']:
 for i,row in enumerate(json.loads((P/f'candidate_{ch}.json').read_text())['rows']):
  if 'candidate'in row:ck('ownactualspectrumidentity '+ch+str(i),H.json_identity(H.spectrum(W,row['candidate']),row['candidate']))
# Exact enumerated actual author records; no glob-based report confusion.
for ch in ['EC','PC','NEUTRON_DECAY']:
 for order in [2,4]:
  for closure in ['empty','LTE','noneq']:
   f=A/'pre_independent_record_repair/author/evidence'/f'pilot_{ch}_{order}_{closure}.json';z=json.loads(f.read_text())['result'];ck('authoractualpilotidentity '+f.name,H.json_identity(H.reconstruction(W,z),z))
for ch in ['EC','PC']:
 for n in [4,8,16,24,32]:
  for first in range(0,n,4):
   f=A/'pre_independent_record_repair/study/raw'/f'{ch}_{n}_{first:02d}.json';z=json.loads(f.read_text());assert z['complete']
   for row in z['rows']:ck('actualincidentnodeidentity '+ch+str(n)+' '+str(row['index']),H.json_identity(H.spectrum(W,row['result']),row['result']))
r=H.reconstruction(W,g['pilots'][0]);pos=lambda x:W.PositiveQuantity(**dataclasses.asdict(x))
# Additional mutations target distinct aggregate/input fields, preserving physical nodes.
for key in ['forward','backward']:
 for index in [0,1]:
  values=list(getattr(r,key));old=values[index];values[index]=Q.positive(old.log+math.log(1.1));reject(key+str(index)+'changed',lambda key=key,values=values:dataclasses.replace(r,**{key:tuple(values)}))
for i in [0,1]:
 vals=list(r.net);v=vals[i];vals[i]=Q.signed(v.log_abs+math.log(1.2),v.sign);reject('signednet'+str(i),lambda vals=vals:dataclasses.replace(r,net=tuple(vals)))
reject('positiveentropy changed',lambda:dataclasses.replace(r,entropy_production=Q.positive(r.entropy_production.log+math.log(2))))
for kwargs in [dict(nodes=r.nodes[::-1]),dict(nodes=r.nodes[:1]),dict(whole_moment_qualified=True),dict(complete_finite_range=False),dict(units=('bad','bad'))]:reject('structural '+str(list(kwargs)),lambda kwargs=kwargs:dataclasses.replace(r,**kwargs))
# Coherent tiny coefficient logs admitted with ordinary None; arbitrary finite logs
# here test REPRESENTATION only, not physical tail accuracy.
for log in [-1000.,-11000.,1000.]:
 q=S.coefficient(log);ck('finite log representation '+str(log),q.log==log and q.value is None and q.support=='positive_support')
for values in [(0.,-1000.,'positive_support','underflow'),(False,None,'exact_kinematic_zero','analytic_empty_particle_hole_support'),(None,float('nan'),'positive_support','underflow')]:reject('invalidquantity '+repr(values),lambda values=values:W.SpectralQuantity(*values))
# Independent AST audit over initial immutable own copy and repaired own copy.
oldast=ast.parse((P/'candidate_copy/src/nmbackend/reference/_weak_spectral_common.py').read_text());newast=ast.parse((P/'repaired_copy/src/nmbackend/reference/_weak_spectral_common.py').read_text());od={n.name:n for n in oldast.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};nd={n.name:n for n in newast.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))};dump=lambda seq:[ast.dump(n,include_attributes=False)for n in seq]
oloop=next(n for n in od['reconstruct'].body if isinstance(n,ast.For));nloop=next(n for n in nd['_assemble_spectral_nodes'].body if isinstance(n,ast.For));ix=next(i for i,n in enumerate(oloop.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='base'for t in n.targets));ck('pernodearithmetic exactAST',dump(oloop.body[ix:])==dump(nloop.body));ck('physicalevaluation exactAST',dump(oloop.body[:ix])==dump(next(n for n in nd['reconstruct'].body if isinstance(n,ast.For)).body))
for name in ['SpectralControls','SpectralQuantity','EnergyRule','coefficient','empty_coefficient','validate','differential_inputs','dos']:ck('unchangedAST '+name,ast.dump(od[name],include_attributes=False)==ast.dump(nd[name],include_attributes=False))
for name in ['_weak_current.py','_weak_two_body.py','_weak_decay.py','_weak_quadrature.py','_weak_spectral_capture.py','_weak_spectral_decay.py','weak_reference.py']:ck('unchangedphysicalbytes '+name,(P/'candidate_copy/src/nmbackend/reference'/name).read_bytes()==(P/'repaired_copy/src/nmbackend/reference'/name).read_bytes())
# Fresh DIFFERENT valid runtime check: EC same previously qualified fine point,
# entire coefficient and finite-range payload exact equality, not merely KMS.
b=W.FreeElectronBath(**g['pilots'][0]['bath']);c=W.SpectralControls(**g['pilots'][0]['controls']);rad=W.Radiation(**g['pilots'][0]['radiation']);rule=W.EnergyRule(**{**g['pilots'][0]['rule'],'nodes':tuple(g['pilots'][0]['rule']['nodes']),'weights':tuple(g['pilots'][0]['rule']['weights'])})
z=W.reconstruct_spectral_moments('EC',b,rad,rule,c);ck('freshECFDpilot bitwise',H.json_identity(z,g['pilots'][0]));(P/'REPAIR_FRESH_EC_PILOT.json').write_text(json.dumps(dataclasses.asdict(z),indent=2)+'\n')
z=W.capture_differential_coefficients('PC',b,4.61,5.897,.99987,c);ck('freshgenuineempty unchanged',z.j.log is None and z.true_chi.log is None and z.j.value==z.true_chi.value==0.)
out=dict(passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],checks=checks,exceptions=exceptions,seconds=time.monotonic()-start,fresh_physical_calls='one EC2nodeFDpilot +one genuineempty PC differential',scope='narrow repair, constructor coherence/sourceidentity; NOT new wholemoment qualification')
(P/'REPAIR_RECHECK_RESULTS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(out['passed'],'PASS',len(out['failed']),'FAIL',out['seconds'],'seconds');print(out['failed'])
