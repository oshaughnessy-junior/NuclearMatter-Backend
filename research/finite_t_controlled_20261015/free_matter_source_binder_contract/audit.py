"""Saved-only source/API/number/energy arithmetic; no scientific imports/calls."""
from pathlib import Path
import json,hashlib,math,ast,datetime
R=Path(__file__).resolve().parent;C=R.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda name:json.loads((C/name).read_text())
checks=[]
def check(name,passed,**evidence):checks.append(dict(name=name,passed=bool(passed),**evidence))
def close(name,a,b,gate=3e-9):
 err=abs(a-b)/max(abs(a),abs(b),1e-300);check(name,err<=gate,error=err,gate=gate,a=a,b=b)
modules={'PR51':'production_lepton_port/staged/src/nmbackend/reference/finite_t_leptons.py','PR52':'production_baryon_port/staged/src/nmbackend/reference/finite_t_baryons.py','PR54':'production_weak_port/staged/src/nmbackend/reference/weak_reference.py'}
remote=json.loads((R/'PUBLISHED_SOURCE_READS.json').read_text());inventory={}
for row in remote:
 data=row['result']['structuredContent']['content'].encode();local=(C/modules[row['label']]).read_bytes()
 check('Published immutable '+row['label']+' entire module bytes',data==local,ref=row['ref'],path=row['path'],sha256=hashlib.sha256(data).hexdigest())
for label,path in modules.items():
 tree=ast.parse((C/path).read_text());inventory[label]={'path':path,'sha256':sha(C/path),'functions':{},'classes':{}}
 for node in tree.body:
  if isinstance(node,ast.FunctionDef) and not node.name.startswith('_'):inventory[label]['functions'][node.name]=ast.unparse(node.args)
  if isinstance(node,ast.ClassDef):inventory[label]['classes'][node.name]={'fields':[{ 'name':x.target.id,'type':ast.unparse(x.annotation),'default':None if x.value is None else ast.unparse(x.value)} for x in node.body if isinstance(x,ast.AnnAssign)],'methods':{x.name:ast.unparse(x.args) for x in node.body if isinstance(x,ast.FunctionDef) and not x.name.startswith('_')}}
forward=load('installed_free_matter_binding/forward_results.json');mapped=[]
for i,row in enumerate(forward['states']):
 s,o=row['new'],row['old'];T,n,Y=s['T'],s['nB'],s['Ye']
 for sp,target in [('neutron',(1-Y)*n),('proton',Y*n)]:close(f'forward{i} {sp} particle inversion',s[sp]['particle_density'],target,1e-9)
 close(f'forward{i} electron NET inversion',s['electron']['net_density'],Y*n,1e-9)
 close(f'forward{i} pair difference',s['electron']['particle_density']-s['electron']['antiparticle_density'],s['electron']['net_density'])
 for sp in ('neutron','proton','electron'):
  for f in ('chemical_potential','energy_density','pressure','entropy_density'):
   close(f'forward{i} independent accepted component {sp} {f}',s[sp][f],o[sp][f])
 mapped.append({'pointer':f'/states/{i}/new','T':T,'nB':n,'Ye':Y,'bath':dict(T=T,mu_n=s['neutron']['chemical_potential'],mu_p=s['proton']['chemical_potential'],mu_e=s['electron']['chemical_potential']),'baryon_etas':[(s[x]['chemical_potential']-s[x]['mass'])/T for x in ('neutron','proton')],'electron_total_eta':s['electron']['chemical_potential']/T})
 for eta in mapped[-1]['baryon_etas']:check(f'forward{i} actual baryon rate admission',-100<=eta<=-5,value=eta)
 check(f'forward{i} actual electron rate admission',abs(mapped[-1]['electron_total_eta'])<=3)
# Only AFTER unperturbed saved-forward checks pass: audit proposed response chain.
check('Saved forward gates precede response arithmetic',all(x['passed'] for x in checks))
w=load('installed_free_matter_binding/weak_fine.json');s=w['state'];T,n,Y=s['T'],s['nB'],s['Ye']
h={}
for sp,dname in [('neutron','dmu_dtemperature_fixed_particle_density'),('proton','dmu_dtemperature_fixed_particle_density'),('electron','dmu_dtemperature_fixed_net_density')]:h[sp]=s[sp]['chemical_potential']-T*s[sp][dname]
close('e_Y fixed nB,T, explicit NETelectron',s['e_Y'],n*(h['proton']-h['neutron']+h['electron']))
close('e_n fixed T,Ye',s['e_n'],(1-Y)*h['neutron']+Y*(h['proton']+h['electron']))
cv=sum(s[x][f] for x,f in [('neutron','cv_fixed_particle_density'),('proton','cv_fixed_particle_density'),('electron','cv_fixed_net_density')])+s['photon_cv'];close('Cv exact fixed nB,Ye chain',s['cv'],cv)
for total,f in [('energy','energy_density'),('pressure','pressure'),('entropy','entropy_density')]:close('Material '+total,s[total],sum(s[x][f] for x in h)+s['photon_'+total])
close('Photons blackbody',s['photon_energy'],math.pi**2*T**4/(15*197.3269804**3))
for field,expected in [('photon_pressure',s['photon_energy']/3),('photon_entropy',4*s['photon_energy']/(3*T)),('photon_cv',4*s['photon_energy']/T)]:close(field,s[field],expected)
events={}
for ch in ('EC','PC','NEUTRON_DECAY'):
 path=f'spectral_band_total_closure/events/{ch}_empty.json';a=load(path);check(ch+' actual saved event complete',a['complete']);events[ch]=a['result']
 for f,val in a['bath'].items():check(ch+' EOS chemical forward mapping '+f,abs(val-w['source_input']['bath'][f])<=1e-11,absolute_difference=abs(val-w['source_input']['bath'][f]))
 key='beta' if ch=='NEUTRON_DECAY' else ch
 for k,f in enumerate(('number','neutrino_energy')):close(ch+' saved event vs accepted binding '+f,events[ch]['forward'][k]['value'],w['source']['channels'][key][f],3e-6)
 check(ch+' inverse structural absence',all(x['value']==0. and x['log'] is None and x['reason']=='absent_initial_neutrino_occupation' for x in events[ch]['backward']))
rate=lambda ch:events[ch]['net'][0]['value'];energy=lambda ch:events[ch]['net'][1]['value']
Rnum=rate('PC')+rate('NEUTRON_DECAY')-rate('EC');Q=sum(energy(ch) for ch in events)
close('Net proton/electron number source',Rnum,w['source']['R']);close('Neutrino energy escape',Q,w['source']['Q'])
close('Ye dot uses particle-count source/nB',Rnum/n,w['Ye_dot'])
Tdot=(((s['energy']+s['pressure'])/n-s['e_n'])*w['ndot']-Q-s['e_Y']*Rnum/n)/s['cv'];close('Rest-inclusive instantaneous Tdot',Tdot,w['T_dot'])
close('Energy loss no extra Q',sum(events[ch]['net'][5]['value'] for ch in events),Q)
check('Default masses differ and require explicit overrides',(939.565421,938.272088,.510998950)!=(939.56542052,938.27208816,.51099895069))
result=dict(kind='Saved arithmetic/API audit ONLY, no binder/physics implementation',new_scientific_calls=0,checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],mapped_saved_states=mapped,instantaneous=dict(R_fm3_s=Rnum,Q_MeV_fm3_s=Q,Ye_dot_s=Rnum/n,T_dot_MeV_s=Tdot),source_current='Same accepted Guo free current, not independent empirical model',whole_spectral_moments_qualified=False)
(R/'API_INVENTORY.json').write_text(json.dumps(inventory,indent=2)+'\n');(R/'EVIDENCE.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k in ['passed','failed','new_scientific_calls','instantaneous']},indent=2))
