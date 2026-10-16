"""Serial isolated heavy-mass benchmark, explicit baseline-before-limit stages."""
from pathlib import Path
import sys,json,hashlib,math,subprocess,time
P=Path(__file__).resolve().parent;ROOT=P.parent;lock=json.loads((P/'LOCK.json').read_text());checks=[]
def ck(n,e,t=3e-6):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def child(scale,profile,method):
 path=P/f'lambda{int(scale)}_{method}_profile{profile}.json'
 if path.exists():raise RuntimeError('No silent resume/overwrite')
 name=path.stem;r=subprocess.run([sys.executable,'-B',str(P/'worker.py'),str(scale),str(profile),method],capture_output=True,text=True,timeout=110)
 (P/(name+'.stdout')).write_text(r.stdout);(P/(name+'.stderr')).write_text(r.stderr)
 if r.returncode:raise RuntimeError('Child failure preserved '+name)
 return json.loads(path.read_text())
def qualify(r):
 scale,profile,method=r['scale'],r['profile'],r['method'];n=lock['state']['nB'];y=lock['state']['Ye'];label=str((scale,profile,method));m=r['masses']
 ck(label+' fixed physical Q',m['actual_Q']-m['Q0'],2e-12)
 for species,mass,target in [('neutron',m['mn'],n*(1-y)),('proton',m['mp'],n*y)]:ck(label+' actualEOSmass '+species,r['EOS'][species]['mass']/mass-1,0);ck(label+' actualEOSdensity '+species,r['EOS'][species]['density']/target-1,1e-10)
 ck(label+' netelectroncharge',r['EOS']['electron']['net_density']/(n*y)-1,1e-10)
 for ch in['EC','PC']:
  ck(label+' actualFDdensity '+ch,r['density_residuals'][ch],3e-8)
  for j,k in enumerate(['number','energy']):ck(label+' positive '+ch+k,0 if r['values'][ch][j]>0 else 1,0)
 ec=r['raw']['EC'];pc=r['raw']['PC'];ev=ec['scenarios']['empty'];pv=pc['sums'][0][0];ck(label+' ECon-shell',ec['max_neutron_mass_residual'],3e-10);ck(label+' ECfourmomentum',ec['max_four_residual']/m['Mweak'],3e-10);ck(label+' ECenergyledger',ev['energy_ledger_scaled'],3e-10);ck(label+' PCfourmom/mass',pc['max_shell'],3e-10);ck(label+' PCenergyledger',(pv[3]-pv[4]-pv[2]+pv[1])/pv[1],3e-10);ck(label+' PCdirectenergyledger',pv[5]/pv[1]-1,3e-10)
def compare(a,b,label):
 for ch in['EC','PC']:
  for j,k in enumerate(['number','energy']):ck(label+ch+k,a['values'][ch][j]/b['values'][ch][j]-1,3e-6)
def baseline():
 if(P/'BASELINE_FORWARD.json').exists():raise RuntimeError('Immutable baseline exists')
 c=json.loads((P/'current_controls.json').read_text())
 if c['failed']:raise RuntimeError('Pointcurrent gate failed')
 start=time.monotonic();rows=[child(1,0,'scalar'),child(1,1,'scalar'),child(1,0,'spinor')]
 for r in rows:qualify(r)
 compare(rows[0],rows[1],'lambda1 profile');compare(rows[2],rows[0],'lambda1 absolute literalspinor vs scalar')
 # Compute same-constant stationary allowed anchor only after forward checks.
 import importlib.util
 sp=importlib.util.spec_from_file_location('heavy_limit_static_1D_reference',ROOT/'weak_rates/static_capture.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S);r=rows[1];T=lock['state']['T'];n=lock['state']['nB'];y=lock['state']['Ye'];allowed={ch:(target*S.target_moments(T,r['bath']['mu_e'],name,q=r['masses']['Q0'])).tolist()for ch,name,target in [('EC','electron_capture',n*y),('PC','positron_capture',n*(1-y))]}
 out=dict(status='BASELINE_FORWARD_PASS'if all(c['passed']for c in checks)else'FAILED_EXCLUDED',checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,allowed=allowed,seconds=time.monotonic()-start);(P/'BASELINE_FORWARD.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'],out['seconds'])
def limit():
 if(P/'results.json').exists():raise RuntimeError('Immutable final evidence')
 b=json.loads((P/'BASELINE_FORWARD.json').read_text())
 if b['status']!='BASELINE_FORWARD_PASS':raise RuntimeError('No heavy-mass interpretation before absolute lambda1 forward gate')
 start=time.monotonic();rows=b['rows'][:2]
 for scale in[2,4,8]:
  a=child(scale,0,'scalar');z=child(scale,1,'scalar');rows.extend([a,z]);qualify(a);qualify(z);compare(a,z,'lambda'+str(scale)+' profile');(P/'partial.json').write_text(json.dumps(dict(status='INCOMPLETE',rows=rows,checks=checks),indent=2)+'\n')
 allowed=b['allowed'];model=[];table=[]
 def mc(name,error,gate):model.append(dict(name=name,error=float(error),gate=gate,passed=bool(math.isfinite(error)and abs(error)<=gate)))
 for r in rows[1::2]:
  e={ch:[v/allowed[ch][j]-1 for j,v in enumerate(r['values'][ch])]for ch in['EC','PC']};table.append(dict(scale=r['scale'],values=r['values'],allowed=allowed,relative_errors=e,bath=r['bath'],masses=r['masses']))
 for ch in['EC','PC']:
  for j,k in enumerate(['number','energy']):
   er=[abs(t['relative_errors'][ch][j])for t in table]
   for i in range(3):
    ratio=er[i+1]/er[i];mc(f'monotone abs error {ch}/{k}/{i}',0 if er[i+1]<er[i]else 1,0);mc(f'1/M ratio within[.25,.75] {ch}/{k}/{i}',max(0.,.25-ratio,ratio-.75),0)
   mc('lambda8 within1percent '+ch+k,er[-1],.01)
 # Charge and lepton EOS are undeformed; moving target distributions are fresh.
 for r in rows:ck('fixed leptonEOS '+str((r['scale'],r['profile'])),r['bath']['mu_e']-b['rows'][0]['bath']['mu_e'],0)
 for p,h in lock['pins'].items():ck('frozen input unchanged '+p,0 if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h else 1,0)
 out=dict(status='AUTHOR_LIMIT_ASSESSMENT_DIFFERENT_REVIEW_REQUIRED',checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],model_checks=model,model_passed=sum(c['passed']for c in model),model_failed=[c for c in model if not c['passed']],baseline_forward=b,table=table,rows=rows,seconds=time.monotonic()-start,combined_physical_calls=9,scope='Artificial common heavy baryon mass limit at fixed Q/T/n/Ye and pointV-A; no physicalmassprediction/uncertainty or uniform claim.');(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'],'model',out['model_passed'],out['model_failed'],'seconds',out['seconds']);print(table)
if __name__=='__main__':
 if sys.argv[1]=='baseline':baseline()
 elif sys.argv[1]=='limit':limit()
 else:raise ValueError('Explicit baseline/limit stage')
