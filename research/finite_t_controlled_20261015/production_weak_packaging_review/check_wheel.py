from pathlib import Path
import sys,json,hashlib,zipfile,math,io,contextlib,runpy,ast,importlib.metadata,time
P=Path(__file__).resolve().parent;A=P.parent/'production_weak_port';wheel=next((P/'wheels').glob('*.whl'));sys.dont_write_bytecode=True;checks=[];start=time.monotonic();sha=lambda b:hashlib.sha256(b).hexdigest()
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
m=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());base=json.loads((P/'PREPARATION.json').read_text());omissions=[]
with zipfile.ZipFile(wheel)as z:
 names=z.namelist()
 for path,h in m['files'].items():
  raw=(P/'package_copy'/path).read_bytes();ck('exact overlay '+path,sha(raw)==h)
  if path.endswith('.py'):ast.parse(raw.decode(),feature_version=(3,9));ck('3.9 grammar '+path,True)
  if path.startswith('src/'):ck('exact wheel module '+path,sha(z.read(path[4:]))==h)
 for r in base['baseline_blobs']:
  name=r['path'][4:]
  if name not in names:omissions.append(r['path']);ck('baseline omission nonPython '+r['path'],not name.endswith('.py'));continue
  raw=z.read(name);ck('baseline unchanged '+r['path'],hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==r['blob'])
 extras=sorted(n for n in names if n.startswith('nmbackend/')and n.endswith('.py')and 'src/'+n not in[r['path']for r in base['baseline_blobs']]);ck('only five new package modules',extras==sorted(p[4:]for p in m['files']if p.startswith('src/')))
 ck('no unmerged finiteT EOS dependencies',not any('finite_t_'+s in n for s in ['rmf','leptons','baryons']for n in names))
 md=z.read('nmbackend-0.0.1.dist-info/METADATA').decode();(P/'wheel_METADATA.txt').write_text(md)
 for line in ['Requires-Python: >=3.9','Requires-Dist: numpy>=1.21','Requires-Dist: scipy>=1.7']:ck('metadata '+line,line in md)
 ck('README discovery','weak_reference'in md)
 ck('source-only docs/example/fixture baselinepolicy',all(n not in names for n in ['docs/weak_reference.md','examples/free_weak_reference.py','tests/data/free_weak_reference.json']))
 (P/'wheel_filelist.json').write_text(json.dumps(names,indent=2)+'\n')
sys.path.insert(0,str(wheel))
import nmbackend
from nmbackend import constants as C
from nmbackend.reference import weak_reference as W,_weak_current as V,_weak_quadrature as Q,_weak_two_body as B,_weak_decay as D
for name,module in [('root',nmbackend),('constants',C),('public',W),('current',V),('quadrature',Q),('two_body',B),('decay',D)]:ck('actualwheel origin '+name,str(module.__file__).startswith(str(wheel)+'/'))
for record in [W.Current,W.FreeElectronBath,W.Radiation,W.Controls,W.MomentResult,Q.PositiveQuantity,Q.SignedQuantity]:ck('immutable '+record.__name__,record.__dataclass_params__.frozen)
ck('physical fixed masses',W.Current().neutron_mass==939.56542052 and W.Current().proton_mass==938.27208816 and W.Current().electron_mass==.51099895069)
ck('no eager export',not hasattr(nmbackend,'channel_moments'))
f=io.StringIO()
with contextlib.redirect_stdout(f):runpy.run_path(str(P/'package_copy/examples/free_weak_reference.py'),run_name='__main__')
(P/'example_stdout.txt').write_text(f.getvalue());lines=f.getvalue().splitlines();ck('exactexample three named channels',len(lines)==3 and [s.split()[0]for s in lines]==list(W.CHANNELS))
for channel,line in zip(W.CHANNELS,lines):
 ck('example positive emitted number '+channel,float(line.split('=')[1].split()[0])>0)
 ck('example gross energy closure '+channel,abs(float(line.split('=')[-1]))<1e-8)
def reject(name,fn,exception=ValueError):
 try:fn()
 except exception:ck(name,True)
 else:ck(name,False)
reject('StageA spectrum unavailable',lambda:W.spectral_coefficients(),NotImplementedError)
bath=W.FreeElectronBath(2.,919.8287578100662,916.8449303084382,.17446333200826397)
reject('unsupported channel',lambda:W.channel_moments('MUON',bath));reject('temperature mismatch',lambda:W.channel_moments('EC',bath,W.Radiation.fd(1.9,0.,0.)))
reject('Boolean bath',lambda:W.FreeElectronBath(True,919.,916.,.1));reject('huge scalar',lambda:W.FreeElectronBath(10**1000,919.,916.,.1));reject('incompatible controls',lambda:W.channel_moments('EC',bath,controls={}))
reject('signed positive falsezero rejected',lambda:Q.SignedQuantity(0.,None,1));reject('signed negative falsezero rejected',lambda:Q.SignedQuantity(0.,None,-1))
ck('research runtime not imported',not any('outputs/nmb-finite-t'in str(getattr(x,'__file__',''))and not str(getattr(x,'__file__','')).startswith(str(wheel)+'/')for n,x in list(sys.modules.items())if x is not None and n!='__main__'))
pins=json.loads((P/'BEFORE_PIN.json').read_text());changed=[r for r,h in pins.items()if sha((A/r).read_bytes())!=h];ck('all current author bytes unchanged',not changed)
out=dict(status='PASS'if all(x['passed']for x in checks)else'FAIL',passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],checks=checks,baseline_packaging_omissions=omissions,wheel_sha256=sha(wheel.read_bytes()),wheel=str(wheel),actual_module_origin=W.__file__,python=sys.version,versions={n:importlib.metadata.version(n)for n in ['numpy','scipy','jax','diffrax','setuptools','wheel']},elapsed_seconds=time.monotonic()-start,physical_calls='exact repository example: three base empty-radiation channels; no new grid',minimum_runtime_executed=False,changed=changed)
(P/'wheel_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['status'],out['passed'],out['failed'],out['elapsed_seconds']);sys.exit(0 if out['status']=='PASS'else 1)
