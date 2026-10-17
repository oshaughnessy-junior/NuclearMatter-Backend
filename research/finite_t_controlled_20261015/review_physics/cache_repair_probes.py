"""Independent synthetic-only cache mismatch probes; no physics integration."""
from pathlib import Path
import sys,hashlib,json,copy,tempfile
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'quantum_energy_moments'
sys.path.insert(0,str(P))
import cache_contract as cache
EXPECTED={'cache_contract.py':'1f40edb4887bfed2052e241f05fc840ea9eb9f78831d5e9e3766c94360357460',
          'run_study.py':'faf527020caf55a7efae542ecf8eedeb070d02ffdd96bc21d48e6b198b7f9fd5'}
def pinned():
 for name,h in EXPECTED.items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
pinned()
m=cache.integrated.slab.matter_state(2.,7e-5,3e-5)
req=cache.fingerprint(m,'nue',outer_order=4,internal_order=4,edges=(0.,.1))
key=cache.digest(req);expected={key:req}
result=dict(flavor='nue',outer_order=4,internal_order=4,internal_tail=40.,length_cm=1e5,
            panels=[dict(lo=0.,hi=.1,values=[123.])],synthetic_marker='NOT a physical integration')
row=cache.record(req,result);checks=[]
def rejected(path,expect=expected):
 try:cache.load(path,expect)
 except (ValueError,KeyError,TypeError):return True
 return False
def add(name,value):checks.append(dict(name=name,passed=bool(value)))
with tempfile.TemporaryDirectory() as td:
 path=Path(td)/'synthetic.json'
 cache.save(path,[row],.1)
 add('exact synthetic replay',cache.load(path,expected)[key]==result)
 wrong=cache.fingerprint(cache.integrated.slab.matter_state(2.01,7e-5,3e-5),'nue',outer_order=4,internal_order=4,edges=(0.,.1))
 cache.save(path,[cache.record(wrong,result)],.1)
 add('wrong actual state with self-consistent metadata rejected',rejected(path))
 mutated=copy.deepcopy(req);mutated['source_sha256']['nr_baryon_eos/free_nucleons.py']='0'*64
 cache.save(path,[cache.record(mutated,result)],.1)
 add('upstream hash mutated and metadata digest recomputed still rejected',rejected(path))
 cache.save(path,[row,cache.record(mutated,result)],.1)
 add('valid leading record followed by stale record rejects whole file',rejected(path))
 cache.save(path,[row,row],.1);add('duplicates rejected',rejected(path))
 cache.save(path,[row],.1);payload=json.loads(path.read_text());payload['records'][0]['result']['synthetic_marker']='corrupted'
 path.write_text(json.dumps(payload));add('payload corruption rejected',rejected(path))
 path.write_text(json.dumps(dict(runs=[result])));add('legacy metadata-free file rejected',rejected(path))
 add('missing cache rejected',rejected(Path(td)/'missing.json'))
pinned()
out=dict(scope='Synthetic cache protocol only; no author cache or physics payload mutated',
         source_sha256=EXPECTED,checks=checks,all_passed=all(c['passed'] for c in checks))
Path(__file__).with_name('cache_repair_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
