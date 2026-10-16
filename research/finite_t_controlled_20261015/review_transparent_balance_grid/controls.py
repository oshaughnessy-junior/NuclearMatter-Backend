from pathlib import Path
import importlib.util,sys,json,math,copy,hashlib
P=Path(__file__).resolve().parent;A=P.parent/'transparent_balance_grid';sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('balance_grid_guard_review_only',A/'grid.py');G=importlib.util.module_from_spec(s);sys.modules[s.name]=G;s.loader.exec_module(G);checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
for args in [(True,1e-7,.3),(float('nan'),1e-7,.3),(1.,float('inf'),.3),(1.,1e-7,False),(2.,1e-7,.3),(1.,1e-7,.049),(1.,1e-7,.901)]:
 try:G.material(*args);good=False
 except ValueError:good=True
 ck('reject material '+repr(args),good)
for value in [0,1,'true',None,float('nan')]:
 for field in ['refined','tail_extended']:
  try:G.evaluate(1.,1e-7,.3,**{field:value});good=False
  except ValueError:good=True
  ck('reject nonliteral '+field+repr(value),good)
acceptance=A/'EC_REVIEW_ACCEPTANCE.json';original=Path.read_text;real=json.loads(acceptance.read_text());material=G.material;calls=[]
def spy(*a,**kw):calls.append(True);raise AssertionError('physical preparation reached')
G.material=spy
for name,modify in [('flag integer',lambda x:x.update(accepted=1)),('flag false',lambda x:x.update(accepted=False)),('source hash',lambda x:x.update(source_sha256='0'*64)),('review hash',lambda x:x.update(review_sha256='0'*64))]:
 fake=copy.deepcopy(real);modify(fake)
 def patched(path,*a,**kw):return json.dumps(fake)if path==acceptance else original(path,*a,**kw)
 Path.read_text=patched
 try:G.evaluate(1.,1e-7,.3);good=False
 except RuntimeError:good=True
 finally:Path.read_text=original
 ck('reject before preparation '+name,good and not calls)
G.material=material
# Mandatory source pin changed in memory; acceptance rejects before EOS.
old=G.PINS['hot_electron_capture_moments/electron_capture.py'];G.PINS['hot_electron_capture_moments/electron_capture.py']='0'*64;G.material=spy
try:G.evaluate(1.,1e-7,.3);good=False
except RuntimeError:good=True
finally:G.PINS['hot_electron_capture_moments/electron_capture.py']=old;G.material=material
ck('changed current source rejects before preparation',good and not calls)
# Loaded local dependency tree must be covered by actual source pins.
pins=set(G.PINS)|set(json.loads((A/'UPSTREAM.json').read_text()));loaded=set()
for module in list(sys.modules.values()):
 f=getattr(module,'__file__',None)
 if not f:continue
 try:r=str(Path(f).resolve().relative_to(P.parent))
 except ValueError:continue
 if r!=str(Path(__file__).resolve().relative_to(P.parent)):loaded.add(r)
ck('all eleven loaded physical files source pinned',not(loaded-pins-{'transparent_balance_grid/grid.py'}))
upstream=A/'UPSTREAM.json';fake=json.loads(upstream.read_text());name='muon_full_current/current.py';fake[name]='0'*64
Path.read_text=lambda path,*a,**kw:json.dumps(fake)if path==upstream else original(path,*a,**kw);G.material=spy
try:G.evaluate(1.,1e-7,.3);good=False
except RuntimeError:good=True
finally:Path.read_text=original;G.material=material
ck('changed transitive current rejects before preparation',good and not calls)
# Independent stoichiometry and every saved separate gross channel moment.
summary=json.loads((A/'summary.json').read_text())
for i in range(3):
 a=json.loads((A/f'state_{i}.json').read_text());r=a['balanced'];channels=r['channels'];RB=channels['beta']['number'];RP=channels['PC']['number'];RE=channels['EC']['number'];net=RB+RP-RE;gross=RB+RP+RE;Q=sum(c['neutrino_energy']for c in channels.values());le=net+RE-RB-RP
 ck(f'root absolute gross {i}',abs(net/gross)<1e-7);ck(f'root net Le {i}',abs(le/gross)<1e-14);ck(f'Q arithmetic {i}',Q==r['escaping_energy']);ck(f'thermostat heat positive {i}',Q/r['nB']>0)
 # Rest-inclusive species material change; no added neutron/proton mass-Q.
 delta=channels['EC']['neutron_energy']-channels['beta']['neutron_energy']-channels['PC']['neutron_energy']+channels['beta']['proton_energy']+channels['PC']['proton_energy']-channels['EC']['proton_energy']+channels['beta']['electron_energy']-channels['PC']['positron_energy']-channels['EC']['electron_energy'];ck(f'independent species energy {i}',abs((delta+Q)/Q)<3e-10)
 ck(f'pair consumption at root {i}',abs((RB-RE)+RP)/gross<1e-7);ck(f'kinetic affinity nonzero {i}',abs(r['chemical_affinity'])>.1)
 ck(f'chemical root actual drift differs {i}',abs(a['chemical_source']['net_material_number_source'])/sum(a['chemical_source']['rates'].values())>1e-3)
 ck(f'all separate paired profiles tested {i}',all(pair['coarse']['refined']is False and pair['refined']['refined']is True for pair in a['rows']))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']]);(P/'controls_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
