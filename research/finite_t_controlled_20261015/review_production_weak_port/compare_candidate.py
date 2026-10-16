from pathlib import Path
import json,sys,time,dataclasses,hashlib,math
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'candidate_copy/src'))
from nmbackend.reference import weak_reference as W
L=json.loads((P/'PREPARATION_LOCK.json').read_text());i=int(sys.argv[1]);ch=sys.argv[2];q=L['baths'][i];b=W.FreeElectronBath(**{k:q[k]for k in['T','mu_n','mu_p','mu_e']});fine=json.loads((P/f'pure_{i}_fine.json').read_text());out=dict(bath=q,channel=ch,profile='candidatebuiltinFine vs originalprelockedindependentFine',rows={},checks=[],complete=False,origins={n:str(m.__file__)for n,m in sys.modules.items()if n.startswith('nmbackend')and getattr(m,'__file__',None)});start=time.monotonic();dest=P/f'candidate_{i}_{ch}.json'
def ck(n,e,g):out['checks'].append(dict(name=n,error=float(e),gate=g,passed=bool(abs(e)<=g)))
def save():
 out['seconds']=time.monotonic()-start;out['passed']=sum(z['passed']for z in out['checks']);out['failed']=[z for z in out['checks']if not z['passed']];dest.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
for case,rr in q['radiation'].items():
 rad=W.Radiation.empty()if rr is None else W.Radiation.fd(**{'temperature':rr['T'],'mu_nue':rr['mu_nue'],'mu_antinue':rr['mu_antinue']})
 z=W.channel_moments(ch,b,rad,W.Controls.fine(ch));out['rows'][case]=dataclasses.asdict(z);ref=fine['channels'][ch]
 if ch=='PC':
  j=list(q['radiation']).index(case);F,B=ref['sums'][j];N=ref['net'][j];S=ref['entropy'][j]
 else:F,B,N,S=[ref['scenarios'][case][k]for k in['F','B','net','entropy']]
 # Actual forward parent density gate first.
 md=dict(z.diagnostics);parent='proton'if ch=='EC'else'neutron';ck(case+' actualgivenmuparentnumber',md['parent_particle_density']/fine['forward'][parent]['number']-1,3e-9)
 for branch,reference in[('forward',F),('backward',B)]:
  for k,(v,r)in enumerate(zip(getattr(z,branch),reference)):
   if r>0:ck(case+branch+str(k),v.value/r-1,3e-6);ck(case+branch+str(k)+'log',v.log-math.log(r),3e-6)
   else:ck(case+branch+str(k)+'exactempty',0 if v.value==0. and v.log is None and v.reason=='absent_initial_neutrino_occupation'else 1,0)
 save()
 if out['failed']:out['stage']='forwardfailure';save();raise ArithmeticError('forwardcounterexamplepreserved')
 gross=F[0]+B[0];gQ=F[1]+B[1]
 for k,v in enumerate(z.net):ck(case+' directnet'+str(k),(v.value-N[k])/(F[k]+B[k]),3e-6)
 ck(case+' energyledger',z.energy_ledger_scaled,3e-9);ck(case+' directenergyledger',z.direct_energy_ledger_scaled,3e-9)
 if rr is not None:
  ck(case+' directentropy',(z.entropy_production.value-S)/gross,3e-6);ck(case+' entropypositive',min(0,z.entropy_production.value)/gross,3e-9)
  ck(case+' eventKMS',z.max_event_kms_error,3e-9)
  if case=='LTE':ck(case+' net/gross',z.net[0].value/gross,2e-10)
 else:ck(case+' noemptyradiationentropy',0 if z.entropy_production is None else 1,0)
 save()
out['complete']=True;out['origins'].update({n:str(m.__file__)for n,m in sys.modules.items()if n.startswith('nmbackend')and getattr(m,'__file__',None)})
for n,h in json.loads((P/'CANDIDATE_PIN.json').read_text())['candidate_files'].items():assert hashlib.sha256((P/'candidate_copy'/n).read_bytes()).hexdigest()==h,n
save();print(i,ch,'PASS',out['passed'],'FAIL',len(out['failed']),'seconds',out['seconds'])
