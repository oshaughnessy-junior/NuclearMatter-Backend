from pathlib import Path
import sys,json,math,hashlib,importlib.util,time
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;s=importlib.util.spec_from_file_location('spectral_tail_only_reference',P/'reference.py');D=importlib.util.module_from_spec(s);sys.modules[s.name]=D;s.loader.exec_module(D);L=D.V.LOCK;H=json.loads((P/'HIGH_LOCK.json').read_text());ch,E,kind=sys.argv[1],float(sys.argv[2]),sys.argv[3];assert kind in['material','outer'];assert(P/'TAIL_LOCK.json').exists();assert any(z['channel']==ch and z['E']==E and z['kind']==kind for z in json.loads((P/'TAIL_LOCK.json').read_text())['jobs'])
if ch=='NEUTRON_DECAY':
 assert kind=='material'and E in L['beta_incident_points'];base=json.loads((P/f'pure_NEUTRON_DECAY_{E}_fine.json'if E==.337 else P/f'high_NEUTRON_DECAY_{E}.json').read_text())['row'];ctrl=dict(L['profiles']['beta_fine']if E==.337 else H['profiles']['beta_high']);ctrl['tail']=95.;st=time.monotonic();row=D.V.B.integral(E,**ctrl);row['logj']=row.pop('j_log');row['logchi']=row.pop('chi_log')
else:
 assert dict(channel=ch,E=E)in H['jobs'];z=json.loads((P/f'high_{ch}_{E}.json').read_text());assert not z['failed'];base=z['row'];ctrl=dict(H['profiles']['capture_high']);ctrl['tail'if kind=='material'else'outer_tail']=95. if kind=='material'else 60.;st=time.monotonic();row=D.integrated(ch,E,**ctrl)
checks=[]
for key in['j','chi']:
 err=math.expm1(row['log'+key]-base['log'+key]);checks.append(dict(name=kind+' onlyextension '+key,error=err,gate=3e-6,passed=abs(err)<=3e-6))
for n,h in L['sourcepins'].items():assert hashlib.sha256((P.parent/n).read_bytes()).hexdigest()==h
out=dict(channel=ch,E=E,kind=kind,controls=ctrl,row=row,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],complete=True,candidate_imported=False,seconds=time.monotonic()-st,scope='Isolatedboundextensiononly, no rigorousinfinitedomainerrorcertificate')
(P/f'tail_{ch}_{E}_{kind}.json').write_text(json.dumps(out,indent=2)+'\n');print(ch,E,kind,'PASS',out['passed'],'FAIL',len(out['failed']),'seconds',out['seconds'])
