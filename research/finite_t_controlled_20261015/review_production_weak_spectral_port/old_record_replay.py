from pathlib import Path
import sys,importlib.util,json
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'candidate_copy/src'))
from nmbackend.reference import weak_reference as W
s=importlib.util.spec_from_file_location('spectral_review_exact_old_record_converter',P/'record_replay_helpers.py');H=importlib.util.module_from_spec(s);s.loader.exec_module(H)
g=json.loads((P/'GUARD_RESULTS.json').read_text());rows=[]
for r in g['witnesses']:
 if r['name'].startswith('REQUIRED'):
  z=H.reconstruction(W,r['accepted'])if r['accepted_type']=='ReconstructionResult'else H.spectrum(W,r['accepted']);rows.append(dict(name=r['name'],old_wrong_acceptance_reproduced=True,exact_json_identity=H.json_identity(z,r['accepted'])))
assert len(rows)==4 and all(r['exact_json_identity']for r in rows)
(P/'OLD_RECORD_REPLAY.json').write_text(json.dumps(dict(kind='ExpectedBADoldacceptance replay; NOTphysicalvalidation/waiver',passed=4,failed=[],rows=rows,physical_calls=0,source='ownimmutableinitialcandidate_copy'),indent=2)+'\n');print('OLD4wrongacceptancesREPRODUCED; no physicalcalls')
