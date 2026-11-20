"""Pure provenance regressions plus actual entry early rejection in a child."""
from pathlib import Path
import copy,json,time
import qualification_contract as c
P=Path(__file__).resolve().parent;start=time.monotonic();checks=[]
expected=c.fingerprint(2.,1e-4,.3);report=json.loads((P/'source_qualification.json').read_text())
def gate(name,value):checks.append(dict(name=name,passed=bool(value)))
def reject(name,payload,want):
 try:c.validate(payload,want)
 except (ValueError,KeyError,TypeError):gate(name,True)
 else:gate(name,False)
gate('current exact qualified source accepted',c.validate(report,expected)['coarse_source']==report['coarse_source'])
for section,key in [('input_state','T'),('input_state','nB'),('input_state','Ye'),('full_fd_state','neutron'),('thermodynamic_closure','u_T'),('environment','numpy'),('controls','internal_order'),('controls','coarse_order'),('controls','fine_order'),('controls','tail'),('controls','edges'),('controls','inventory')]:
 want=copy.deepcopy(expected);want[section][key]='changed';reject('reject changed '+section+'.'+key,report,want)
for file in expected['source_sha256']:
 want=copy.deepcopy(expected);want['source_sha256'][file]='0'*64;reject('reject changed source '+file,report,want)
payload=copy.deepcopy(report);payload['coarse_source']['moments']['nue']['number_fm3_s']*=2;reject('reject corrupt physical moments',payload,expected)
payload=copy.deepcopy(report);payload['coarse_source']['Ye']+=.01;payload['qualified_payload_sha256']=c.digest({k:v for k,v in payload.items() if k not in ('qualification_schema','qualification_fingerprint','qualification_fingerprint_sha256','qualified_payload_sha256')});reject('reject rehashed wrong result state',payload,expected)
legacy=json.loads((P/'pre_qualification_contract_20261015T1402/source_qualification.json').read_text());reject('reject missing legacy provenance',legacy,expected)
# Current functions/source bytes remain unchanged throughout this regression.
gate('fingerprint stable across checks',expected==c.fingerprint(2.,1e-4,.3))
result=dict(passed=all(x['passed'] for x in checks),checks=checks,runtime_seconds=time.monotonic()-start,physics='No trajectory or additional physical rate calculation performed')
(P/'qualification_contract_results.json').write_text(json.dumps(result,indent=2)+'\n');print(sum(x['passed'] for x in checks),'/',len(checks),result['runtime_seconds']);raise SystemExit(0 if result['passed'] else 1)
