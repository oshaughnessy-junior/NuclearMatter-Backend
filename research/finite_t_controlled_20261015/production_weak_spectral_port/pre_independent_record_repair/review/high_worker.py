from pathlib import Path
import sys,json,importlib.util,math,time,hashlib
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;s=importlib.util.spec_from_file_location('spectral_high_locked_reference',P/'reference.py');D=importlib.util.module_from_spec(s);sys.modules[s.name]=D;s.loader.exec_module(D);H=json.loads((P/'HIGH_LOCK.json').read_text());ch=sys.argv[1];E=float(sys.argv[2]);assert dict(channel=ch,E=E)in H['jobs'];st=time.monotonic()
if ch=='NEUTRON_DECAY':
 row=D.V.B.integral(E,**H['profiles']['beta_high']);row['logj']=row.pop('j_log');row['logchi']=row.pop('chi_log');row['E']=E;row['channel']=ch
else:row=D.integrated(ch,E,**H['profiles']['capture_high'])
fine=json.loads((P/f'pure_{ch}_{E}_fine.json').read_text())['row'];checks=[]
for key in['j','chi']:
 er=math.expm1(fine['log'+key]-row['log'+key]);checks.append(dict(name=key+' fine/highrefinement',error=er,gate=3e-6,passed=abs(er)<=3e-6))
for n,h in D.V.LOCK['sourcepins'].items():assert hashlib.sha256((P.parent/n).read_bytes()).hexdigest()==h,n
out=dict(row=row,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],complete=True,candidate_imported=False,seconds=time.monotonic()-st,scope='Oneextraindependentreferencequad rung; originalcoarsegatesunchanged')
(P/f'high_{ch}_{E}.json').write_text(json.dumps(out,indent=2)+'\n');print(ch,E,'HIGH',out['passed'],'FAIL',len(out['failed']),'seconds',out['seconds'])
