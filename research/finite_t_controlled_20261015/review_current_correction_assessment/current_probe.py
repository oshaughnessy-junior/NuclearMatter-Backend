from pathlib import Path
import ast,math,json,sys,importlib.util,hashlib
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;A=R/'current_correction_assessment';sys.dont_write_bytecode=True
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
C=load('different_F2_direct_trace',R/'muon_full_current/current.py');E=load('different_F2_spinors',R/'relativistic_capture_oracle/events.py');K=load('different_F2_frozen_config',A/'configured_currents.py')
checks=[];rows=[]
def ck(name,error,gate=3e-10):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(math.isfinite(error)and abs(error)<=gate)))
assert sha(A/'configured_currents.py')=='d745784b53dbb1f8c37d6f0e1a8d67a22f95adadc98def254a6bcbbe5c756fbd'
assert sha(A/'MANIFEST.json')=='df5f59ec94248841e7cc2505776a66e50cfdd9feead131a8686cda8c13728165'
for rel,h in json.loads((A/'MANIFEST.json').read_text())['files'].items():assert sha(A/rel)==h,rel
# Regenerate expected ASTs, but never author source or configured artifacts.
class Normalize(ast.NodeTransformer):
 def visit_Attribute(self,node):
  node=self.generic_visit(node)
  if isinstance(node.value,ast.Name)and node.value.id=='E'and node.attr in ['GF','VUD']:return ast.Name(id=node.attr,ctx=ast.Load())
  return node
for channel,path,original in [('ec','hot_electron_capture_moments/electron_capture.py','scalar_amplitude'),('pc','review_pc_moments/reference.py','amplitude')]:
 fn=next(n for n in ast.parse((R/path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==original)
 fn=Normalize().visit(fn)
 for off in [False,True]:
  expected=ast.parse(ast.unparse(fn)).body[0];expected.name=channel+('_F2zero'if off else'_full')
  assigns=[n for n in ast.walk(expected)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='f'for t in n.targets)]
  assert len(assigns)==1
  if off:assigns[0].value=ast.Constant(value=0.)
  actual=next(n for n in ast.parse((A/'configured_currents.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name==expected.name)
  ck('exact source AST '+expected.name,0 if ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False)else 1,0)

def spinor_gordon(ch,p,ff):
 ma,mb=ch.initial_mass,ch.final_mass;M=(ma+mb)/2;f1,ga,f2,gp=ff;q=p['pb']-p['pa'];psum=p['pa']+p['pb']
 vertices=np.array([(f1+f2)*E.G[u]-f2*psum[u]/(2*M)*E.I-ga*E.G[u]@E.G5-gp*q[u]/M*E.G5 for u in range(4)])[None,:,:,:]
 ua=E.spinors(p['pa'][None,:],ma);ub=E.spinors(p['pb'][None,:],mb);uk=E.spinors(p['k'][None,:],0,ch.antineutrino);ul=E.spinors(p['l'][None,:],K.ME,ch.antineutrino);left=E.G@(E.I-E.G5)
 J=np.einsum('nai,ab,nubc,ncj->nuij',ub.conj(),E.G[0],vertices,ua).reshape(1,4,4)
 if ch.antineutrino:L=np.einsum('nai,ab,ubc,ncj->nuij',uk.conj(),E.G[0],left,ul).reshape(1,4,4)
 else:L=np.einsum('nai,ab,ubc,ncj->nuij',ul.conj(),E.G[0],left,uk).reshape(1,4,4)
 amp=np.einsum('nui,u,nuj->nij',L,E.MET,J)
 return E.GF**2*E.VUD**2/8*np.sum(abs(amp)**2)
for anti in [False,True]:
 ch=C.V.Channel(K.MP if anti else K.MN,K.MN if anti else K.MP,K.ME,antineutrino=anti)
 for energy,cosine in [(6.3,-.44),(23.1,.36),(71.2,-.82)]:
  rest=C.V.momenta(ch,energy,cosine)
  for moving in [False,True]:
   p={key:(E.boost(val[None,:],np.array([[.13,-.08,.05]]))[0]if moving and key in ['k','pa','l','pb']else val)for key,val in rest.items()}
   q=p['pb']-p['pa'];t=C.V.dot(q,q);den=(1-t/(4*K.MW**2))*(1-t/840**2)**2
   full=( (1-4.706*t/(4*K.MW**2))/den,1.2723/(1-t/1000**2)**2,3.706/den,0.)
   full=(*full[:3],2*K.MW**2*full[1]/(139.57**2-t));off=(full[0],full[1],0.,full[3]);args=[p[k][None,:]for k in ['k','pa','l','pb']]
   for mode,ff in [('full',full),('F2zero',off)]:
    scalar=float(getattr(K,('pc_'if anti else'ec_')+mode)(*args)[0]);trace=C.trace(ch,p,ff)/2;spin=float(spinor_gordon(ch,p,ff))
    tag=str((anti,energy,moving,mode));ck('Gordon spin sum vs trace '+tag,spin/trace-1);ck('configured scalar vs independent spin '+tag,scalar/spin-1)
    ck('positive amplitude '+tag,0 if scalar>0 else 1,0)
    vs=C.vertices(q,K.MW,ff);rev=C.vertices(-q,K.MW,ff);total=[sum(vs[k][u]for k in vs)for u in range(4)];reverse=[sum(rev[k][u]for k in rev)for u in range(4)]
    ck('adjoint reverse q '+tag,max(float(np.max(abs(C.V.adjoint(total[u])-reverse[u])))for u in range(4)),3e-13)
    magnetic=np.array(vs['F']);scale=max(1.,float(np.max(abs(magnetic))))*max(1.,float(max(abs(q))))
    ck('Pauli transverse '+tag,float(np.max(abs(sum(E.MET[u]*q[u]*magnetic[u]for u in range(4)))))/scale,3e-13)
    rows.append(dict(anti=anti,energy=energy,cosine=cosine,moving=moving,mode=mode,scalar=scalar,trace=trace,spin_gordon=spin,t=t,form_factors=ff))
   fullv=C.vertices(q,K.MW,full);offv=C.vertices(q,K.MW,off)
   for part in ['V','A','P']:ck('unchanged '+part+str((anti,energy,moving)),float(np.max(abs(np.array(fullv[part])-np.array(offv[part])))),0)
   ck('off Pauli zero '+str((anti,energy,moving)),float(np.max(abs(np.array(offv['F'])))),0)
   ck('changing xi everywhere is a different F1 '+str((anti,energy,moving)),0 if abs(full[0]-(1-t/(4*K.MW**2))/den)>1e-10 else 1,0)
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,rows=rows);(P/'current_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed']);assert not out['failed']
