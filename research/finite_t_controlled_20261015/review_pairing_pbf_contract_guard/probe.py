"""Independent bounded recheck of the positive-subnormal input exclusion only."""
from pathlib import Path
import importlib.util,sys,json,math,hashlib,ast,warnings,time
import numpy as np
from scipy.special import zeta
from scipy.integrate import quad
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'pairing_pbf_contract';OLD=A/'pre_admission_repair';PREV=ROOT/'review_pairing_pbf_contract';sys.dont_write_bytecode=True
start=time.monotonic();checks=[];old_failures=[];rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snap():return {str(p.relative_to(ROOT)):sha(p)for folder in [A,PREV]for p in folder.rglob('*')if p.is_file()}
before=snap();(P/'before.json').write_text(json.dumps(before,indent=2)+'\n')
def ck(n,e,g=0):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
def reject(n,f):
 try:f();ck(n,1)
 except ValueError as e:checks.append(dict(name=n,passed=True,rejection=str(e)))
 except Exception as e:checks.append(dict(name=n,passed=False,rejection=type(e).__name__+':'+str(e)))
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
ck('commissioned repaired source SHA',int(sha(A/'quasiparticle.py')!='06f6db3fb69f2fe50042a15d5c129f151354561c3263e4cc976e7cf869a419d0'))
ck('commissioned current manifest SHA',int(sha(A/'MANIFEST.json')!='56b2ac889f605bef332c5a59894aa4f7fca5100521bfc5ad5511e58be2294a7f'))
for rel,h in json.loads((A/'MANIFEST.json').read_text()).items():ck('current manifest '+rel,int(sha(A/rel)!=h))
for rel,h in json.loads((A/'PRE_ADMISSION_REPAIR_PIN.json').read_text()).items():
 ck('old preserved '+rel,int(sha(OLD/rel)!=h))
 ck('independent prior archive '+rel,int(sha(PREV/'reviewed_initial_author'/rel)!=h))
N=load('pbf_admission_repaired_functiononly',A/'quasiparticle.py');O=load('pbf_admission_original_functiononly',OLD/'quasiparticle.py')
# Literal scientific AST remains identical after deleting only exact new guards/docstrings.
def normalized(path,strip):
 tree=ast.parse(path.read_text());removed=[]
 for f in tree.body:
  if isinstance(f,ast.FunctionDef):
   if isinstance(f.body[0],ast.Expr)and isinstance(f.body[0].value,ast.Constant)and isinstance(f.body[0].value.value,str):f.body.pop(0)
   if strip and f.name in ['log_F_A','log_F_B']:
    expected=ast.dump(ast.parse('if v < np.finfo(float).tiny: raise ValueError("Subnormal positive gap input unsupported; require v>=float64 tiny or exact zero")').body[0],include_attributes=False)
    found=[node for node in f.body if ast.dump(node,include_attributes=False)==expected]
    ck('exact one admission guard '+f.name,len(found)-1)
    f.body=[node for node in f.body if node not in found];removed+=found
 return ast.dump(tree,include_attributes=False)
ck('scientific AST identity after only two guards/docstrings',int(normalized(A/'quasiparticle.py',True)!=normalized(OLD/'quasiparticle.py',False)))
# Analytic small-gap coefficient and separate direct-x quadrature anchor.
C=24*((1-2**-4)*zeta(5,1)-(1-2**-3)*zeta(4,1))
Cquad=quad(lambda x:x**4*math.exp(-2*x)/(1+math.exp(-x))**2,0,100,epsabs=1e-14,epsrel=2e-13)[0]
ck('small gap coefficient independent direct integral',Cquad/C-1,3e-13)
tiny=float(np.finfo(float).tiny);sub=float(np.nextafter(0.,1.));below=float(np.nextafter(tiny,0.));above=float(np.nextafter(tiny,math.inf))
for v in [1e-320,sub]:
 with warnings.catch_warnings(record=True)as ws:
  warnings.simplefilter('always');lf=O.log_F_B(v)
 expected=2*math.log(v)+math.log(2*Cquad);error=math.expm1(lf-expected)
 old_failures.append(dict(name='retained original positive-subnormal FB accuracy '+repr(v),passed=bool(abs(error)<=3e-10),error=error,gate=3e-10,v=v,v_hex=v.hex(),logFB=lf,analytic_logFB=expected,warnings=[str(w.message)for w in ws]))
 ck('original FB failure independently reproduced '+repr(v),int(abs(error)<=3e-10))
 for name in ['log_F_A','log_F_B']:reject('new rejects old failed input '+name+' '+v.hex(),lambda v=v,name=name:getattr(N,name)(v))
for name in ['log_F_A','log_F_B']:
 reject('largest subnormal rejected '+name,lambda name=name:getattr(N,name)(below))
 for v in [tiny,above]:
  lf=getattr(N,name)(v);factor=1 if name=='log_F_A'else 2;expected=2*math.log(v)+math.log(factor*Cquad)
  ck('exact normal boundary accuracy '+name+' '+v.hex(),math.expm1(lf-expected),3e-10)
  rows.append(dict(kind='boundary',function=name,v=v,v_hex=v.hex(),log=lf,represented=N.represented(lf)))
# Including both signs of exact zero and selected normal values covering underflow.
normal=[0.,-0.,tiny,above,1e-307,1e-280,1e-250,1e-200,1e-163,.017,.43,2.7,9.3,37.,127.,300.,499.,500.]
for v in normal:
 for name in ['log_F_A','log_F_B']:
  a=getattr(O,name)(v);b=getattr(N,name)(v);ck('bitwise log compatibility '+name+' '+v.hex(),int(a.hex()!=b.hex()))
  ck('representation compatibility '+name+' '+v.hex(),int(O.represented(a)!=N.represented(b)))
  rows.append(dict(kind='compatibility',function=name,v=v,v_hex=v.hex(),old_log_hex=a.hex(),new_log_hex=b.hex(),new_log=b if math.isfinite(b)else '-inf',represented=N.represented(b)))
  if v==0:ck('exact signed-zero retained '+name+' '+v.hex(),int(N.represented(b)!=dict(log_value=None,value=0.,status='exact_zero')))
for name,v in [('log_F_A',1000.),('log_F_B',500.),('log_F_A',1e-200),('log_F_B',1e-200),('log_F_A',tiny),('log_F_B',tiny)]:
 a=getattr(O,name)(v);b=getattr(N,name)(v);r=N.represented(b)
 ck('ordinary output underflow finite log '+name+' '+repr(v),int(not(math.isfinite(b)and r['value']is None and r['status']=='positive_below_float64_representation')))
 ck('deep or small ordinary output bitwise unchanged '+name+' '+repr(v),int(a.hex()!=b.hex()))
for name,upper in [('log_F_A',1000.),('log_F_B',500.)]:
 for v in [True,False,np.bool_(True),None,'1',complex(1),-1.,-tiny,-sub,float('nan'),float('inf'),-float('inf'),float(np.nextafter(upper,math.inf))]:reject('invalid-input unchanged '+name+' '+repr(v),lambda name=name,v=v:getattr(N,name)(v))
for x in [float('nan'),float('inf')]:reject('invalid represented log '+repr(x),lambda x=x:N.represented(x))
# All prior numerical failures and endpoint reports remain byte-exact, not reclassified.
ck('two old failures retained as failures',sum(not c['passed']for c in old_failures)-2)
ck('all author and prior review bytes unchanged',int(before!=snap()))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],retained_original_accuracy_failures=old_failures,rows=rows,coefficient=dict(analytic=C,independent_quad=Cquad),boundaries=dict(smallest_subnormal=sub,largest_subnormal=below,tiny=tiny,above=above),seconds=time.monotonic()-start)
(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],len(out['failed']),'repair checks; retained original FAIL',len(old_failures),'seconds',out['seconds']);print(out['failed']);print([(r['error'],r['warnings'])for r in old_failures])
