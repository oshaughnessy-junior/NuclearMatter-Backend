from pathlib import Path
import json,sys,hashlib,time,dataclasses
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'candidate_copy/src'))
from nmbackend.reference import weak_reference as W
L=json.loads((P/'PREPARATION_LOCK.json').read_text());checks=[];witness=[]
def ck(n,e,g=0):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(abs(e)<=g)))
for sign in[-1,1]:
 try:z=W.SignedQuantity(0.0,None,sign);witness.append(dict(input=dict(value=0.,log_abs=None,sign=sign),accepted=dataclasses.asdict(z)))
 except ValueError:witness.append(dict(sign=sign,rejected=True))
ck('signedzero falsely admits both signs',len([z for z in witness if'accepted'in z]),0)
b=W.FreeElectronBath(**{k:L['baths'][0][k]for k in['T','mu_n','mu_p','mu_e']})
for ch in['EC','PC','NEUTRON_DECAY']:
 st=time.monotonic()
 try:W.channel_moments(ch,b,controls=W.Controls(radial_order=4,lepton_order=4,polar_order=4,azimuth_order=4,rest_order=4,parent_order=4,orient_order=4,decay_azimuth_order=4,budget_seconds=1e-12))
 except TimeoutError:ck('actualtimeout '+ch,0)
 else:ck('actualtimeout '+ch,1)
 ck('boundedtimeout '+ch,0 if time.monotonic()-st<120 else 1)
for i in[0,1]:
 for case in L['baths'][i]['radiation']:
  zz={ch:json.loads((P/f'candidate_{i}_{ch}.json').read_text())['rows'][case]for ch in['EC','PC','NEUTRON_DECAY']}
  EC,PC,BE=[zz[ch]for ch in['EC','PC','NEUTRON_DECAY']];ec,pc,be=[[q['value']for q in z['net']]for z in[EC,PC,BE]];gR=sum(z['forward'][0]['value']+z['backward'][0]['value']for z in zz.values());gQ=sum(z['forward'][1]['value']+z['backward'][1]['value']for z in zz.values())
  dn=ec[0]-pc[0]-be[0];dp=be[0]+pc[0]-ec[0];de=be[0]-ec[0];dpos=-pc[0];dnu=ec[0];da=pc[0]+be[0]
  ck(f'{i}{case} baryon',(dn+dp)/gR,3e-9);ck(f'{i}{case} charge',(dp-de+dpos)/gR,3e-9);ck(f'{i}{case} Le',(de-dpos+dnu-da)/gR,3e-9)
  dEn=ec[4]-pc[4]-be[4];dEp=pc[3]+be[3]-ec[3];dEl=be[2]-ec[2]-pc[2];Q=be[1]+pc[1]+ec[1];ck(f'{i}{case} speciesenergy',(dEn+dEp+dEl+Q)/gQ,3e-9)
  for ch,z in zz.items():
   st=dict(z['particle_stoichiometry']);ck(f'{i}{case}{ch} stoichiometrybaryon',st['neutron']+st['proton']);ck(f'{i}{case}{ch} stoichiometrycharge',st['proton']-st['electron_net']);ck(f'{i}{case}{ch} stoichiometryLe',st['electron_net']+st.get('nue',0)-st.get('antinue',0))
  if case=='empty':ck(f'{i}{case} positiveescape',0 if Q>0 else 1)
# Verify all original author and pure-preparation files after imports and calculations.
pin=json.loads((P/'CANDIDATE_PIN.json').read_text());A=P.parent/'production_weak_port';h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n,v in pin['author_files'].items():ck('unchangedauthor '+n,0 if h(A/n)==v else 1)
for n,v in json.loads((P/'PREPARATION_MANIFEST.json').read_text())['files'].items():ck('unchangedpure '+n,0 if h(P/n)==v else 1)
for n,v in pin['candidate_files'].items():ck('unchangedowncopy '+n,0 if h(P/'candidate_copy'/n)==v else 1)
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,falsezero_witnesses=witness,scope='No physical failure; signed record admission issue',operational_deviations=['Bath1EC was launched while bath0decay previous poll still reported running; overlap two thread1 jobs, both bounded and complete. No resulting physical failure or rerun.','First four channel launches used single fresh05:53:39quota check within two minutes rather than separate check perjob; subsequent expensive jobs guarded individually; account/window remained exact28percent.'])
(P/'final_checks.json').write_text(json.dumps(out,indent=2)+'\n');print('FINAL',out['passed'],'FAIL',len(out['failed']));print(out['failed'])
