from pathlib import Path
import sys,hashlib,json,importlib.util,dataclasses,math,ast,shutil
import numpy as np
P=Path(__file__).resolve().parent;A=P.parent/'production_weak_port';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[];details=[]
def ck(n,e,g=0):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(abs(e)<=g)))
def reject(n,fn):
 try:fn()
 except ValueError:ck(n,0)
 except Exception as e:ck(n+' wrongexception'+type(e).__name__,1)
 else:ck(n,1)
for n,v in{'CANDIDATE_MANIFEST.json':'80f0b4207003353b8bf59f1357c28baedbf85ccb06d0a85b9289bceb798a174e','MANIFEST.json':'347ee290afa7eb4d177b92c2ee65056c90cea2a388d13d74ac4e5f5e8f8c6cd7','AUTHOR_REPORT.md':'f8c9cca81315df39b55321b79f3d804ccc1faa8063414cd150a35db504a448f7'}.items():assert h(A/n)==v,n
M=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());old=json.loads((P/'PINNED_CANDIDATE_MANIFEST.json').read_text());ck('same11paths',0 if set(M['files'])==set(old['files'])and len(M['files'])==11 else 1)
for n,v in json.loads((A/'MANIFEST.json').read_text())['sha256'].items():assert h(A/n)==v,n
for n,v in json.loads((P/'INITIAL_REVIEW_MANIFEST.json').read_text())['files'].items():ck('initialreviewunchanged '+n,0 if h(P/n)==v else 1)
for n,v in old['files'].items():
 if n not in['tests/test_weak_reference_moments.py','src/nmbackend/reference/_weak_quadrature.py']:ck('exactunchangedcandidate '+n,0 if M['files'][n]==v else 1)
 for where in['staged','test_mirror']:ck('newmanifest '+where+n,0 if h(A/where/n)==M['files'][n]else 1)
shutil.copytree(P/'candidate_copy',P/'repaired_copy',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'),dirs_exist_ok=True)
for n in M['files']:shutil.copyfile(A/'staged'/n,P/'repaired_copy'/n)
paths=[P/'candidate_copy/src/nmbackend/reference/_weak_quadrature.py',P/'repaired_copy/src/nmbackend/reference/_weak_quadrature.py'];mods=[]
for i,f in enumerate(paths):
 s=importlib.util.spec_from_file_location('weak_guard_recheck_'+str(i),f);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);mods.append(m)
O,N=mods
for sign in[-1,1]:
 z=O.SignedQuantity(0.,None,sign);ck('oldfalsezerowitness '+str(sign),0 if z.value==0. and z.log_abs is None and z.sign==sign else 1);reject('newrejectoldwitness '+str(sign),lambda sign=sign:N.SignedQuantity(0.,None,sign))
 for log in[None,True,False,np.bool_(True),float('nan'),float('inf'),-float('inf'),10**10000,'0',[],1j]:reject('nonzeroLogguard '+str(sign)+type(log).__name__+str(len(checks)),lambda log=log,sign=sign:N.SignedQuantity(0.,log,sign))
for z in[-1000.,-746.,-745.,-744.,-700.,-10.,0.,1.,700.,710.,1000.]:
 for sign in[-1,1]:
  a,b=O.signed(z,sign),N.signed(z,sign);ck('validsignedbitwise '+str((z,sign)),0 if dataclasses.asdict(a)==dataclasses.asdict(b)else 1)
for z in[None,-1000.,-745.,0.,1000.]:ck('positivehelperbitwise '+str(z),0 if dataclasses.asdict(O.positive(z))==dataclasses.asdict(N.positive(z))else 1)
ck('exactcancellationzero',0 if dataclasses.asdict(N.SignedQuantity(0.,None,0))==dict(value=0.,log_abs=None,sign=0)else 1)
count=0
for f in P.glob('candidate_[01]_*.json'):
 for name,z in json.loads(f.read_text())['rows'].items():
  for rec in z['net']+[z['entropy_production']]if z['entropy_production']is not None else z['net']:
   a,b=O.SignedQuantity(**rec),N.SignedQuantity(**rec);ck('savedactualsigned '+f.stem+name+str(count),0 if dataclasses.asdict(a)==dataclasses.asdict(b)==rec else 1);count+=1
# Strip ONLY new guard statement to show original entire scientific/helper AST remains identical.
t1,t2=[ast.parse(f.read_text())for f in paths];cls=next(n for n in t2.body if isinstance(n,ast.ClassDef)and n.name=='SignedQuantity');post=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='__post_init__');branch=next(n for n in post.body if isinstance(n,ast.If)and n.orelse);newguard=branch.orelse[0];ck('guardexactfinite_log',0 if isinstance(newguard,ast.Expr)and isinstance(newguard.value,ast.Call)and getattr(newguard.value.func,'id','')=='finite_real'else 1);branch.orelse.pop(0);ck('wholeASTidenticalafterguardremoval',0 if ast.dump(t1,include_attributes=False)==ast.dump(t2,include_attributes=False)else 1)
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,actual_saved_signed_records=count,old_helper=h(paths[0]),new_helper=h(paths[1]),manifest=M,source_snapshot_pinned=True)
(P/'repair_recheck.json').write_text(json.dumps(out,indent=2)+'\n');print('RECHECK',out['passed'],'FAIL',len(out['failed']),'actualrecords',count);print(out['failed'])
