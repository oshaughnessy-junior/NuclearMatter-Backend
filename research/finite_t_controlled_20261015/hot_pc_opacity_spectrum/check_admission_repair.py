"""Post-execution consumer repair tests; never regenerate physical node files."""
from pathlib import Path
import sys,importlib.util,json,math,hashlib,copy,contextlib,io
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True

def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
new=load('PC_repaired_consumer_only',P/'total.py')
# Execute preserved consumer text with original file location, so unchanged physics
# modules and raw-data paths retain their original identities. No driver main call.
old=type(sys)('PC_preserved_consumer_only');old.__file__=str(P/'total.py');sys.modules[old.__name__]=old;exec(compile((P/'pre_admission_repair/total.py').read_text(),str(P/'pre_admission_repair/total.py'),'exec'),old.__dict__)
pin=json.loads((P/'PRE_ADMISSION_REPAIR_PIN.json').read_text());checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
def rejected(m,c,row,b,meta):
 try:m.validate_row(c,row,b,meta);return False
 except (ValueError,RuntimeError,ArithmeticError,TypeError):return True
b,meta=new.S.matched();allrows={}
for rule,count in [('coarse',1),('middle',3),('fine',4)]:
 c=new.contract(rule);rows=[r for i in range(count)for r in json.loads((P/f'total_{rule}_{i}.json').read_text())['rows']];allrows[rule]=rows
 for row in rows:ck(f'actual unchanged {rule} node{row["index"]}',not rejected(new,c,row,b,meta))
c=new.contract('fine');base=allrows['fine'][12]
def recompute(row):
 v=row['spectrum'];logs=[v[k]['log']for k in ['j','chi']];v['log_kms_actual']=logs[0]-logs[1]if all(x is not None for x in logs)else None;v['relaxation']=sum(v[k]['value']for k in ['j','chi'])if all(v[k]['representable']for k in ['j','chi'])else None
badrows=[]
for keys in [('j',),('chi',),('j','chi')]:
 row=copy.deepcopy(base)
 for k in keys:row['spectrum'][k]=new.S.represented(None)
 recompute(row);ck('reject falsezero '+str(keys),rejected(new,c,row,b,meta));badrows.append(row)
ck('exact old counterexample reproduced',not rejected(old,c,badrows[-1],b,meta))
for k in ['j','chi']:
 for bad in [None,float('nan'),float('inf'),float('-inf'),True,False,'-100']:
  row=copy.deepcopy(base);row['spectrum'][k]['log']=bad;ck(f'reject {k} invalid log {bad!r}',rejected(new,c,row,b,meta))
 for field,bad in [('value',0.),('value',False),('value',True),('value',float('nan')),('value',float('inf')),('representable',1),('representable',0),('reason','empty_support')]:
  row=copy.deepcopy(base);row['spectrum'][k][field]=bad;ck(f'reject {k} {field} {bad!r}',rejected(new,c,row,b,meta))
# Both finite logs unrepresented: keep independent chemistry relation and flags.
row=copy.deepcopy(base);d=row['spectrum']['log_kms_expected'];row['spectrum']['j']=new.S.represented(-13836.);row['spectrum']['chi']=new.S.represented(-13836.-d);recompute(row)
ck('finite log ordinary underflow retained',not rejected(new,c,row,b,meta));ck('underflow value is None, no floor',all(row['spectrum'][k]['value']is None and row['spectrum'][k]['reason']=='underflow'for k in ['j','chi']))
ck('differential empty-support helper unchanged',new.S.represented(None)==dict(value=0.,log=None,representable=True,reason='empty_support'))
# Capture ONLY the combine output write in memory: no raw/results overwrite.
def aggregate(module):
 captured={};original=Path.write_text
 def capture(path,data,*a,**kw):
  if path!=P/'total_results.json':raise RuntimeError('Unexpected output attempt')
  captured['data']=data;return len(data)
 Path.write_text=capture
 try:
  with contextlib.redirect_stdout(io.StringIO()):module.combine()
 finally:Path.write_text=original
 return captured['data']
a=aggregate(old);btext=aggregate(new);ck('old/new complete aggregate bitwise identical',a==btext);ck('original frozen aggregate bitwise identical',btext==(P/'total_results.json').read_text())
for n,h in pin['original_files'].items():
 ck('preserved snapshot '+n,hashlib.sha256((P/'pre_admission_repair'/n).read_bytes()).hexdigest()==h)
 if n not in ['total.py','MANIFEST.json','RESEARCH_NOTE.md','CHECKPOINT.json']:ck('original artifact unchanged '+n,hashlib.sha256((P/n).read_bytes()).hexdigest()==h)
for n,h in pin['independent_initial_review_files'].items():ck('initial review unchanged '+n,hashlib.sha256((P.parent/'review_hot_pc_opacity_spectrum'/n).read_bytes()).hexdigest()==h)
ck('spectrum code hash unchanged',hashlib.sha256((P/'spectrum.py').read_bytes()).hexdigest()=='5fbc8fc21fc6b3b26bf43331ebe04b31bf66145b1f7bf51c977dfa47e5f78b0b')
out=dict(status='AUTHOR_REPAIR_TESTS_NOT_INDEPENDENT_ACCEPTANCE',checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],preserved_counterexample=badrows[-1],valid_synthetic_underflow=row,aggregate_sha256=hashlib.sha256(btext.encode()).hexdigest())
(P/'admission_repair_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
