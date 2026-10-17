from pathlib import Path
import sys,json,dataclasses,time
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'repaired_copy/src'))
from nmbackend.reference import weak_reference as W
q=json.loads((P/'PREPARATION_LOCK.json').read_text())['baths'][0];b=W.FreeElectronBath(**{k:q[k]for k in['T','mu_n','mu_p','mu_e']});old=json.loads((P/'candidate_0_EC.json').read_text());out={'checks':[],'rows':{},'complete':False};st=time.monotonic()
for case in['empty','LTE']:
 rr=q['radiation'][case];rad=W.Radiation.empty()if rr is None else W.Radiation.fd(rr['T'],rr['mu_nue'],rr['mu_antinue']);z=W.channel_moments('EC',b,rad,W.Controls.fine('EC'));row=json.loads(json.dumps(dataclasses.asdict(z)));out['rows'][case]=row;out['checks'].append(dict(name=case+' exactcompleteAPIrecord',passed=row==old['rows'][case]));(P/'repaired_api.json').write_text(json.dumps(out,indent=2)+'\n')
out.update(complete=True,seconds=time.monotonic()-st,passed=sum(c['passed']for c in out['checks']),failed=[c for c in out['checks']if not c['passed']],origin=W.__file__);(P/'repaired_api.json').write_text(json.dumps(out,indent=2)+'\n');print('REPAIRED API',out['passed'],'FAIL',len(out['failed']),'seconds',out['seconds'])
