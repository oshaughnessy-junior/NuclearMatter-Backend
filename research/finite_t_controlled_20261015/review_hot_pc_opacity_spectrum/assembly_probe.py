from pathlib import Path
import sys,importlib.util,json,math,hashlib,copy
from decimal import Decimal,localcontext
P=Path(__file__).resolve().parent;A=P.parent/'hot_pc_opacity_spectrum';sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('review_PC_pure_assembly',A/'total.py');M=importlib.util.module_from_spec(sp);sys.modules[sp.name]=M;sp.loader.exec_module(M)
b,meta=M.S.matched();checks=[]
def ck(n,e,t=3e-14):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
manifest=json.loads((A/'MANIFEST.json').read_text())
for name,h in manifest['files'].items():ck('frozen manifest '+name,0 if hashlib.sha256((A/name).read_bytes()).hexdigest()==h else 1,0)
c=M.contract('fine');rows=[r for i in range(4)for r in json.loads((A/f'total_fine_{i}.json').read_text())['rows']];ref=json.loads((A/'FRESH_EVENT_REFERENCE.json').read_text());saved=json.loads((A/'total_results.json').read_text())['groups']['fine']['total'];sums={};mu=b.mu_n-b.mu_p-b.mu_e
ck('64 exact node indices',0 if [r['index']for r in rows]==list(range(64))else 1,0)
for row in rows:M.validate_row(c,row,b,meta)
with localcontext()as ctx:
 ctx.prec=75;D=lambda x:Decimal(str(x));pi=D(math.pi);den=2*pi*pi*D(M.S.HC)**3
 for si,(name,mur)in enumerate([('empty',None),('LTE',mu),('noneq',mu-.8)]):
  pair={k:[Decimal(0),Decimal(0)]for k in ['F','B']}
  for row in rows:
   E=D(row['E']);f=Decimal(0)if mur is None else 1/(1+((E-D(mur))/D(b.T)).exp());base=D(row['weight'])*E*E/den
   for key,coeff,occ in [('F','j',1-f),('B','chi',f)]:
    q=base*D(row['spectrum'][coeff]['log']).exp()*occ
    pair[key][0]+=q;pair[key][1]+=q*E
  sums[name]={k:list(map(float,v))for k,v in pair.items()}
  for di,k in enumerate(['F','B']):
   for j in range(2):
    v=sums[name][k][j];want=saved[name][k][j];ck(f'Decimal saved {name}/{k}/{j}',v/want-1 if want else v)
    want=ref['sums'][si][di][j];ck(f'saved spectral vs actual CM {name}/{k}/{j}',v/want-1 if want else v,3e-6)
net={n:dict(R=v['F'][0]-v['B'][0],Q=v['F'][1]-v['B'][1])for n,v in sums.items()}
ck('LTE number zero relative gross',net['LTE']['R']/sums['LTE']['F'][0],3e-12);ck('LTE energy zero relative gross',net['LTE']['Q']/sums['LTE']['F'][1],3e-12)
entropy=net['noneq']['R']*.8/b.T;ck('noneq total entropy positive',0 if entropy>=0 else 1,0)
# A zero log is physically unavailable for this finite-T fully integrated PC bath.
row=copy.deepcopy(rows[12]);old=copy.deepcopy(row)
for k in ['j','chi']:row['spectrum'][k]=M.S.represented(None)
row['spectrum']['relaxation']=0.;row['spectrum']['log_kms_actual']=None
try:M.validate_row(c,row,b,meta);false_zero_rejected=False
except (ValueError,RuntimeError,ArithmeticError):false_zero_rejected=True
checks.append(dict(name='reject synthetic false exact-zero integrated j and chi',passed=false_zero_rejected,expected='reject',actual='rejected'if false_zero_rejected else 'ACCEPTED_FALSE_ZERO'))
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],decimal_sums=sums,net=net,noneq_entropy=entropy,false_zero_original=old,false_zero_mutation=row,qualification='Saved-node arithmetic is distinct from fresh physical quadrature; no frozen file mutated.')
(P/'assembly_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
