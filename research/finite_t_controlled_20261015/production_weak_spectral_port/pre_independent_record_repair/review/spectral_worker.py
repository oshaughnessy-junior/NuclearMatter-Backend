from pathlib import Path
import importlib.util,sys,json,math,time,hashlib
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;s=importlib.util.spec_from_file_location('spectral_prelocked_independent_reference',P/'reference.py');D=importlib.util.module_from_spec(s);sys.modules[s.name]=D;s.loader.exec_module(D);L=D.V.LOCK;stage,idx,profile=sys.argv[1],int(sys.argv[2]),sys.argv[3];assert profile in['coarse','fine'];q=json.loads((P/'point_results.json').read_text());assert q['stage']=='COMPLETE'and not q['failed'];A=json.loads((P/'AUTHOR_LOCK_ADMISSION.json').read_text());assert hashlib.sha256((P.parent/'production_weak_spectral_port/LOCK.json').read_bytes()).hexdigest()==A['author_lock_sha256'];st=time.monotonic()
if stage=='capture':
 point=L['capture_incident_points'][idx];row=D.integrated(point['channel'],point['E'],**L['profiles']['capture_'+profile]);name=f"pure_{point['channel']}_{point['E']}_{profile}.json"
else:
 assert stage=='beta';E=L['beta_incident_points'][idx];row=D.V.B.integral(E,**L['profiles']['beta_'+profile]);row['logj']=row.pop('j_log');row['logchi']=row.pop('chi_log');row['E']=E;row['channel']='NEUTRON_DECAY';name=f'pure_NEUTRON_DECAY_{E}_{profile}.json'
checks=[]
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
for key in['logj','logchi']:ck('actualpositivefinitelog '+key,0 if math.isfinite(row[key])else 1,0)
mu=D.b.mu_p+D.b.mu_e-D.b.mu_n;mu=mu if row['channel']=='EC'else-mu;ck('KMSafterdirectactualproducts',row['logj']-row['logchi']+(row['E']-mu)/D.b.T,3e-10)
if stage=='beta':ck('fixedlabenergyandledger',row['max_energy_residual']/D.MN,3e-10)
for rel,h in L['sourcepins'].items():ck('sourceunchanged '+rel,0 if hashlib.sha256((P.parent/rel).read_bytes()).hexdigest()==h else 1,0)
out=dict(stage=stage,profile=profile,row=row,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],complete=True,candidate_imported=False,seconds=time.monotonic()-st,qualification='Finite selectednumericalreferenceprofile; no infiniterange/transport/domaincertificate')
(P/name).write_text(json.dumps(out,indent=2)+'\n');print(name,'PASS',out['passed'],'FAIL',len(out['failed']),'seconds',out['seconds'])
