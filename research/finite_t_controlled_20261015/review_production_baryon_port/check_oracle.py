from pathlib import Path
import importlib.util,sys,json,math
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('independent_baryon_momentum_only',P/'oracle.py');M=importlib.util.module_from_spec(s);sys.modules[s.name]=M;s.loader.exec_module(M)
lock=json.loads((P/'VALIDATION_LOCK.json').read_text());rows=[];checks=[]
def ck(n,e,t):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for i,p in enumerate(lock['states']):
 st=M.density(p['T'],p['n'],p['mass'])if p['kind']=='density'else M.momentum(p['T'],p['mu'],p['mass']);rows.append(st);tail=M.momentum(st['T'],st['mu'],st['mass'],100.)
 for k in st:
  if k not in['T','mu','mass']:ck('momentumtail '+str((i,k)),st[k]/tail[k]-1 if tail[k]else st[k],3e-10)
 ck('Euler bulk '+str(i),(st['energy']+st['pressure']-st['mu']*st['number']-st['T']*st['entropy'])/st['energy'],3e-12)
 ck('Cvvariance positive '+str(i),0 if st['cv']>0 else 1,0)
 if(st['mu']-st['mass'])/st['T']<0:
  b=M.fugacity(st['T'],st['mu'],st['mass'])
  for k in b:ck('independentBessel '+str((i,k)),st[k]/b[k]-1,3e-10)
out=dict(candidate_imported=False,rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']]);(P/'oracle_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
