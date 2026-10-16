"""Fresh public finite-rule calls versus explicitly labelled own postprocessing."""
from pathlib import Path
import sys,json,dataclasses,time
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import postprocess_successor as P
P.verify();sys.path.insert(0,str(R.parent/'production_weak_spectral_port/test_mirror/src'))
from nmbackend.reference import weak_reference as W
ch,name=sys.argv[1:];locked=P.LOCK;rad=next(v for v in locked['radiation']if v['name']==name)
if name not in ('empty','LTE'):raise ValueError('Only locked actual API comparisons')
b=W.FreeElectronBath(**locked['bath']);c=W.SpectralControls(**locked['controls']);lo,hi=locked['channels'][ch]
radiation=W.Radiation.empty()if name=='empty'else W.Radiation.fd(rad['T'],rad['mu_nue'],rad['mu_antinue'])
start=time.monotonic();r=W.reconstruct_spectral_moments(ch,b,radiation,W.EnergyRule.gauss(lo,hi,4),c);P.verify()
ours=next(v for v in P.process(ch,4)if v['radiation']['name']==name);checks=[]
def ck(label,error):checks.append(dict(name=label,error=float(error),gate=3e-12,passed=bool(abs(error)<=3e-12)))
for i in(0,1):
 gross=ours['forward'][i]['value']+ours['backward'][i]['value']
 ck(f'F{i}',r.forward[i].value/ours['forward'][i]['value']-1)
 ck(f'B{i}',(r.backward[i].value-ours['backward'][i]['value'])/gross)
 ck(f'NET{i}',(r.net[i].value-ours['net'][i]['value'])/gross)
if name=='LTE':ck('entropy',(r.entropy_production.value-ours['entropy']['value'])/(ours['forward'][0]['value']+ours['backward'][0]['value']))
source_rows,_=P.collect(ch,4)
for i,(node,saved)in enumerate(zip(r.nodes,source_rows)):
 for key in('j','true_chi'):ck(f'node{i}{key}',getattr(node,key).log-saved['result'][key]['log'])
assert r.complete_finite_range and not r.whole_moment_qualified
out=dict(kind='ACTUAL public API call vs own in-memory radiation postprocessing',channel=ch,radiation=name,checks=checks,passed=sum(v['passed']for v in checks),failed=[v for v in checks if not v['passed']],actual_public_result=dataclasses.asdict(r),seconds=time.monotonic()-start,source_pin_sha256=P.H(R/'SUCCESSOR_SOURCE_PIN.json'),accepted_successor_admission_sha256=P.H(R/'SUCCESSOR_SOURCE_ADMISSION.json'),public_execution_candidate=P.ADMISSION['repaired_candidate_manifest'],reference_execution_group=ours['execution_source_group'],lock_sha256=P.H(R/'LOCK.json'))
(R/'raw_successor'/f'public_agreement_{ch}_{name}.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(ch,name,out['passed'],'PASS',len(out['failed']),'FAIL',out['seconds'])
