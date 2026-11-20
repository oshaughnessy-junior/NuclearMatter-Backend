"""Physical current modules never imported in this parent orchestration."""
from pathlib import Path
import sys,json,hashlib,time,subprocess,math,os
P=Path(__file__).resolve().parent;R=P.parent;A=R/'current_correction_assessment';start=time.monotonic()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=json.loads((A/'LOCK.json').read_text());author=json.loads((A/'results.json').read_text())
def inventory():
 paths=list(A.rglob('*'))+[R/rel for rel in lock['pins']]+list((R/'free_capture_moment_fit').rglob('*'))
 return{str(p.relative_to(R)):sha(p)for p in paths if p.is_file()and '__pycache__'not in str(p)}
before=inventory();checks=[];data={}
def ck(n,e,t):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
ref=json.loads((R/'transparent_balance_grid/state_1.json').read_text())['balanced']['channels']
assert json.loads((P/'current_results.json').read_text())['failed']==[]
for mode in ['full','F2zero']:
 profiles=[]
 for fine in [0,1]:
  r=subprocess.run([sys.executable,'-B',str(P/'thermal_worker.py'),mode,str(fine)],capture_output=True,text=True,timeout=100)
  (P/f'thermal_{mode}_{fine}.stdout').write_text(r.stdout);(P/f'thermal_{mode}_{fine}.stderr').write_text(r.stderr)
  assert r.returncode==0,r.stderr
  value=json.loads((P/f'thermal_{mode}_{fine}.json').read_text());profiles.append(value)
  for name in ['mu_n','mu_p','mu_e']:ck(f'{mode}/{fine} independent EOS {name}',value['bath'][name]-value['accepted_bath'][name],3e-10)
  for channel in ['EC','PC']:
   ck(f'{mode}/{fine} physical FDdensity {channel}',value['density'][channel],3e-8)
   row=value['channels'][channel]
   for field,number in row.items():
    ck(f'{mode}/{fine} positive {channel}/{field}',0 if number>0 else 1,0)
    if mode=='full':ck(f'BASELINE forward {fine}/{channel}/{field}',number/ref[channel][field]-1,3e-6)
   energy=(row['proton_energy']+row['electron_energy']-row['neutron_energy']if channel=='EC'else row['neutron_energy']+row['positron_energy']-row['proton_energy'])
   ck(f'{mode}/{fine} species energy {channel}',energy/row['neutrino_energy']-1,3e-10);ck(f'{mode}/{fine} directloss {channel}',row['material_energy_loss']/row['neutrino_energy']-1,3e-10)
  assert all(c['passed']for c in checks),'Forward gate before any F2zero child'
 for channel in ['EC','PC']:
  for field,val in profiles[1]['channels'][channel].items():ck(f'{mode} moment refine {channel}/{field}',profiles[0]['channels'][channel][field]/val-1,3e-6)
 assert all(c['passed']for c in checks),'Profile gate before correction'
 data[mode]=profiles
comparisons={}
for channel in ['EC','PC']:
 comparisons[channel]={}
 for i,field in enumerate(['number','neutrino_energy']):
  full=data['full'][-1]['channels'][channel][field];off=data['F2zero'][-1]['channels'][channel][field];aa=author['comparison'][channel][i];removal=off/full-1;allowed=aa['allowed']
  ck('fresh removal versus author '+channel+'/'+field,removal-aa['F2_removal_fraction'],3e-6)
  ck('ratio multiplicative identity '+channel+'/'+field,(allowed/off)*(off/full)/(allowed/full)-1,3e-14)
  comparisons[channel][field]=dict(full=full,F2zero=off,removal_fraction=removal,allowed_over_off=allowed/off-1,allowed_over_full=allowed/full-1)
ck('same physical EOS across current switch',0 if data['full'][0]['eos']==data['F2zero'][0]['eos']else 1,0)
ck('four distinct child PIDs',0 if len({v['pid']for g in data.values()for v in g})==4 else 1,0)
ck('same orchestrator parent PID',0 if all(v['parent_pid']==os.getpid()for g in data.values()for v in g)else 1,0)
ck('no current/EOS phase imports in orchestration parent',0 if not any('current.py'in str(getattr(m,'__file__',''))or'electron_capture.py'in str(getattr(m,'__file__',''))for m in sys.modules.values())else 1,0)
after=inventory();ck('author,upstream,fitdefault bytes unchanged',0 if before==after else 1,0)
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,comparison=comparisons,profiles=data,before=before,after=after,seconds=time.monotonic()-start)
(P/'thermal_results.json').write_text(json.dumps(out,indent=2)+'\n');print(dict(passed=out['passed'],failed=out['failed'],comparison=comparisons,seconds=out['seconds']));assert not out['failed']
