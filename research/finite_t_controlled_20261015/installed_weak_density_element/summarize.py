from pathlib import Path
import json,math,sys
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import element as E
rows=[json.loads((P/f'trajectory_{n}.json').read_text())for n in E.L['steps']];ends={p:json.loads((P/f'endpoint_{p}.json').read_text())for p in E.L['endpoint_profiles']};checks=[];comparisons=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
g=E.L['gates']
for a,b in zip(rows[:-1],rows[1:]):
 tag=str(a['steps'])+'to'+str(b['steps']);ck('T absolute '+tag,a['final']['T']-b['final']['T'],g['time_abs_T_MeV']);ck('Ye absolute '+tag,a['final']['Ye']-b['final']['Ye'],g['time_abs_Ye']);qa=a['conserved']['E_nue']+a['conserved']['E_antinue'];qb=b['conserved']['E_nue']+b['conserved']['E_antinue'];ck('escape relative '+tag,qa/qb-1,g['time_escape_relative']);comparisons.append(dict(steps=[a['steps'],b['steps']],dT=a['final']['T']-b['final']['T'],dYe=a['final']['Ye']-b['final']['Ye'],escape_relative=qa/qb-1))
for a,b in [('base','fine'),('fine','tail')]:
 for ch in ['EC','PC','beta']:
  for field,v in ends[a]['source']['channels'][ch].items():ck('actual endpoint '+a+'to'+b+' '+ch+field,v/ends[b]['source']['channels'][ch][field]-1,g['endpoint_source_refine_relative'])
initial=rows[-1]['endpoints'][0]['source'];ef=ends['fine']['source'];change=dict(R=ef['R']/initial['R']-1,Q=ef['Q']/initial['Q']-1)
ck('sources actually change',0 if abs(change['R'])>.01 and abs(change['Q'])>.005 else 1,0)
for steps in [True,0,-2,3]:
 try:E.run(steps)
 except ValueError:ck('invalid locked steps '+str(steps),0,0)
 else:ck('invalid locked steps '+str(steps),1,0)
E.verify();coarse=[dict(steps=z['steps'],failed=z['failed'])for z in rows if z['failed']];counts={str(z['steps']):dict(passed=z['passed'],failed=len(z['failed']))for z in rows};counts.update({p:dict(passed=z['passed'],failed=len(z['failed']))for p,z in ends.items()});summary=dict(status='PASS selected finer trajectories with preserved coarse failures'if not any(not c['passed']for c in checks)and not rows[-1]['failed']else'FAIL',checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],current_coarse_failures=coarse,trial_counts=counts,total_passed=sum(z['passed']for z in rows)+sum(z['passed']for z in ends.values())+sum(c['passed']for c in checks),total_failed=sum(len(z['failed'])for z in rows)+sum(len(z['failed'])for z in ends.values())+sum(not c['passed']for c in checks),comparison=comparisons,source_change=change,final=rows[-1]['final'],final_accumulators=rows[-1]['conserved'])
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary['status'],summary['total_passed'],summary['total_failed'],summary['failed'])
