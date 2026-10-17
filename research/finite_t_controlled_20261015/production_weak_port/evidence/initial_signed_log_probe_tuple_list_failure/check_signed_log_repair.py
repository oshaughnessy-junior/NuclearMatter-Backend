from pathlib import Path
import sys,importlib.util,json,math,dataclasses,hashlib,ast
P=Path(__file__).resolve().parents[1];R=P.parent;sys.dont_write_bytecode=True
checks=[]
def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
old=load('weak_port_original_signed_log_quad',P/'pre_signed_log_repair/staged/src/nmbackend/reference/_weak_quadrature.py')
new=load('weak_port_repaired_signed_log_quad',P/'staged/src/nmbackend/reference/_weak_quadrature.py')
witnesses=[]
for sign in [-1,1]:
 obj=old.SignedQuantity(0.,None,sign);witnesses.append(dataclasses.asdict(obj));check('preserved old falsezero accepted '+str(sign),obj.value==0 and obj.sign==sign and obj.log_abs is None)
 try:new.SignedQuantity(0.,None,sign);reject=False
 except ValueError:reject=True
 check('repaired exact original witness rejected '+str(sign),reject)
 for z in [None,True,False,float('nan'),float('inf'),-float('inf'),'0',10**1000]:
  try:new.SignedQuantity(0.,z,sign);reject=False
  except ValueError:reject=True
  check('invalid nonzero log '+str((sign,str(z)[:20])),reject)
 for log in [-1000.,-745.,-700.,-3.2,0.,1.7,700.,1000.]:
  a=old.signed(log,sign);b=new.signed(log,sign);check('valid signed bitwise '+str((sign,log)),dataclasses.asdict(a)==dataclasses.asdict(b))
check('exactzero unchanged',dataclasses.asdict(old.signed(0.,0))==dataclasses.asdict(new.signed(0.,0)))
# Every saved actual net-moment record is re-admitted identically, no rate grid replay.
count=0
for q in(P/'evidence').glob('candidate_*.json'):
 data=json.loads(q.read_text())
 for rows in data['profiles'].values():
  for row in rows.values():
   for v in row['net']+[row['entropy_production']]if row['entropy_production']is not None else row['net']:
    a=old.SignedQuantity(**v);b=new.SignedQuantity(**v);check('actual signed record '+str(count),dataclasses.asdict(a)==dataclasses.asdict(b));count+=1
# The AST outside SignedQuantity is byte-equivalent scientific code.
a=ast.parse((P/'pre_signed_log_repair/staged/src/nmbackend/reference/_weak_quadrature.py').read_text());b=ast.parse((P/'staged/src/nmbackend/reference/_weak_quadrature.py').read_text())
for t in[a,b]:t.body=[n for n in t.body if not isinstance(n,ast.ClassDef)or n.name!='SignedQuantity']
check('all other quadrature AST unchanged',ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False))
# Fresh real-package public API at two existing qualified scenarios, exact outputs.
sys.path.insert(0,str(P/'test_mirror/src'));from nmbackend.reference import weak_reference as W
saved=json.loads((P/'evidence/candidate_0_EC.json').read_text())['profiles']['base'];bath=W.FreeElectronBath(**saved['empty']['bath'])
for name,rad in[('empty',W.Radiation.empty()),('LTE',W.Radiation.fd(bath.T,bath.equilibrium_mu_nue,bath.equilibrium_mu_antinue))]:
 actual=dataclasses.asdict(W.channel_moments('EC',bath,rad));check('fresh public actual output bitwise '+name,actual==saved[name])
pin=json.loads((P/'PRE_SIGNED_LOG_REPAIR_PIN.json').read_text());h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
check('whole original snapshot exact',all(h(P/'pre_signed_log_repair'/k)==v for k,v in pin['original_full_source_evidence'].items()))
check('whole original review exact',all(h(R/k)==v for k,v in pin['initial_review_files'].items()))
check('60 original scientific pins unchanged',all(h(R/k)==v for k,v in json.loads((P/'evidence/SOURCE_PIN.json').read_text()).items()))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),original_falsezero_witnesses=witnesses,saved_valid_signed_records=count,new_physical_calls=2,scope='helper admission only; no physical/current/phase/default/gate changes')
(P/'evidence/signed_log_repair_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='checks'}))
