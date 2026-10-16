from pathlib import Path
import sys,json,hashlib,zipfile,math,io,contextlib,runpy,ast,importlib.metadata,time
from dataclasses import replace,asdict
P=Path(__file__).resolve().parent;A=P.parent/'production_weak_spectral_port';R=P.parent/'review_production_weak_spectral_port';wheel=next((P/'wheels').glob('*.whl'));checks=[];start=time.monotonic();sys.dont_write_bytecode=True;sha=lambda b:hashlib.sha256(b).hexdigest()
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
def reject(n,fn,exc=ValueError):
 try:fn()
 except exc:ck(n,True)
 else:ck(n,False)
m=json.loads((A/'CANDIDATE_MANIFEST.json').read_text());base=json.loads((P/'BASE_PIN.json').read_text());prep=json.loads((P/'CANDIDATE_PREPARATION.json').read_text());omissions=[]
with zipfile.ZipFile(wheel)as z:
 names=z.namelist()
 for p,h in m['files'].items():
  raw=(P/'package_copy'/p).read_bytes();ck('overlay exact '+p,sha(raw)==h)
  if p.endswith('.py'):ast.parse(raw.decode(),feature_version=(3,9));ck('3.9grammar '+p,True)
  if p.startswith('src/'):ck('wheel exact overlay '+p,sha(z.read(p[4:]))==h)
 for p,r in prep['unchanged_base_files'].items():
  ck('unchanged file '+p,sha((P/'package_copy'/p).read_bytes())==r['sha256'])
  if p.startswith('src/'):
   if p[4:]not in names:omissions.append(p);ck('baseline omitted nonPython '+p,not p.endswith('.py'))
   else:ck('unchangedwheel '+p,sha(z.read(p[4:]))==r['sha256'])
 ck('only three new package modules',sorted(n for n in names if n.startswith('nmbackend/')and n.endswith('.py')and 'src/'+n not in base['files'])==sorted(p[4:]for p in m['new']if p.startswith('src/')))
 ck('no unmerged EOS modules',not any('finite_t_'+s in n for s in ['leptons','baryons','rmf']for n in names))
 md=z.read('nmbackend-0.0.1.dist-info/METADATA').decode();(P/'wheel_METADATA.txt').write_text(md)
 for line in ['Requires-Python: >=3.9','Requires-Dist: numpy>=1.21','Requires-Dist: scipy>=1.7']:ck('metadata '+line,line in md)
 ck('README spectral discovery','spectral'in md)
 ck('docs examples fixtures repositorypolicy',all(p not in names for p in m['files']if p.startswith(('docs/','tests/','examples/'))))
 (P/'wheel_filelist.json').write_text(json.dumps(names,indent=2)+'\n')
sys.path.insert(0,str(wheel))
import nmbackend
from nmbackend import constants as C
from nmbackend.reference import weak_reference as W,_weak_current as V,_weak_quadrature as Q,_weak_two_body as T,_weak_decay as D,_weak_spectral_common as S,_weak_spectral_capture as SC,_weak_spectral_decay as SD
for n,mod in [('root',nmbackend),('constants',C),('public',W),('current',V),('totalhelper',Q),('capturetotal',T),('decaytotal',D),('spectralcommon',S),('spectralcapture',SC),('spectraldecay',SD)]:ck('actualwheel origin '+n,str(mod.__file__).startswith(str(wheel)+'/'))
bath=W.FreeElectronBath(*S.FIXED_BATH);controls=W.SpectralControls()
reject('legacy noexplicitcontrols stub',lambda:W.spectral_coefficients('EC',bath,5.),NotImplementedError)
reject('missing requiredargs explicitsignature',lambda:W.spectral_coefficients(),TypeError)
for obj in [W.SpectralControls,W.SpectralQuantity,W.SpectralResult,W.EnergyRule,W.ReconstructionResult]:ck('immutable '+obj.__name__,obj.__dataclass_params__.frozen)
ck('no eager new root export',not hasattr(nmbackend,'spectral_coefficients'))
# Exact example selected spectra: existing saved points, not a newphysicsgrid.
oldargv=sys.argv;sys.argv=['free_weak_reference.py','--spectra'];out=io.StringIO()
with contextlib.redirect_stdout(out):runpy.run_path(str(P/'package_copy/examples/free_weak_reference.py'),run_name='__main__')
sys.argv=oldargv;(P/'example_spectra_stdout.txt').write_text(out.getvalue());lines=out.getvalue().splitlines();ck('example threechannels',len(lines)==3 and [x.split()[0]for x in lines]==list(W.CHANNELS))
for ch,line in zip(W.CHANNELS,lines):ck('example positive jtruechi '+ch,all(float(line.split(label)[1].split()[0])>0 for label in ['j [s^-1]=','TRUE chi [s^-1]=']))
rule=W.EnergyRule.gauss(2.,3.,2);pilot=W.reconstruct_spectral_moments('EC',bath,W.Radiation.empty(),rule,controls)
(P/'finite_pilot.json').write_text(json.dumps(asdict(pilot),indent=2)+'\n')
ck('finite2nodecoverage',pilot.complete_finite_range and not pilot.whole_moment_qualified and len(pilot.nodes)==2)
ck('empty radiation inverseSOURCEzero',all(x.value==0 for x in pilot.backward)and all(n.true_chi.value>0 for n in pilot.nodes))
node=pilot.nodes[0];ck('truechi vsrelax',abs(node.relaxation.value/(node.j.value+node.true_chi.value)-1)<2e-14);ck('cm inverse units',abs(node.absorption_cm_inverse.value/(node.true_chi.value/S.LIGHT_CM_S)-1)<2e-14);ck('km inverse units',abs(node.absorption_km_inverse.value/node.absorption_cm_inverse.value/1e5-1)<2e-14)
for label,fn in [('stale radiation',lambda:replace(pilot,radiation=W.Radiation.fd(2.,0.,0.))),('stale positive aggregate',lambda:replace(pilot,forward=(Q.positive(pilot.forward[0].log+math.log(2)),pilot.forward[1]))),('missing node',lambda:replace(pilot,nodes=pilot.nodes[:-1])),('whole flag',lambda:replace(pilot,whole_moment_qualified=True)),('incomplete flag',lambda:replace(pilot,complete_finite_range=False)),('stale controls',lambda:replace(pilot,controls=W.SpectralControls.fine())),('integratedfalseempty',lambda:replace(node,j=S.empty_coefficient()))]:reject(label,fn)
near=replace(bath,mu_e=bath.mu_e+1e-10);reject('nearstate rejected',lambda:W.spectral_coefficients('EC',near,5.,controls));reject('BooleanE',lambda:W.spectral_coefficients('EC',bath,True,controls));reject('decaycapture measure reject',lambda:W.capture_differential_coefficients('NEUTRON_DECAY',bath,.8,1.,0.,controls));reject('signedlog regression',lambda:Q.SignedQuantity(0.,None,1))
positive=W.capture_differential_coefficients('EC',bath,4.61,5.897,.27,controls)
reject('original falseempty witness',lambda:replace(positive,j=S.empty_coefficient(),true_chi=S.empty_coefficient(),geometry=('analytic_empty_particle_hole_support',)))
empty=W.capture_differential_coefficients('PC',bath,4.61,5.897,.99987,controls);ck('genuine analytic empty remains valid',empty.j.value==empty.true_chi.value==0. and replace(empty)==empty)
under=S.coefficient(-14000.);ck('finitelog underflow retained',under.value is None and under.log==-14000. and under.support=='positive_support')
ck('no research runtime imports',not any('outputs/nmb-finite-t'in str(getattr(x,'__file__',''))and not str(getattr(x,'__file__','')).startswith(str(wheel)+'/')for n,x in list(sys.modules.items())if x is not None and n!='__main__'))
pins=json.loads((P/'BEFORE_PIN.json').read_text());changed=[]
for kind,root in [('author',A),('numeric_review',R)]:changed +=[kind+'/'+p for p,h in pins[kind].items()if sha((root/p).read_bytes())!=h]
ck('all author/numeric pins unchanged',not changed)
result=dict(status='PASS'if all(c['passed']for c in checks)else'FAIL',passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,omissions=omissions,wheel_sha256=sha(wheel.read_bytes()),wheel=str(wheel),elapsed_seconds=time.monotonic()-start,python=sys.version,versions={n:importlib.metadata.version(n)for n in ['numpy','scipy','jax','diffrax','setuptools','wheel']},actual_origin=W.__file__,changed=changed,whole_RQ_grid_executed=False,minimum_runtime_executed=False)
(P/'wheel_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'],result['passed'],result['failed'],result['elapsed_seconds']);sys.exit(0 if result['status']=='PASS'else 1)
