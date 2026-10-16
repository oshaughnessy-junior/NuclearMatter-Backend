"""Independent saved-only binder design audit: stdlib, no scientific imports."""
from pathlib import Path
import json,hashlib,ast,math
P=Path(__file__).resolve().parent;B=P.parent;A=B/'free_matter_source_binder_contract';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks=[]
def ck(name,error=0.,gate=0.):checks.append(dict(name=name,error=float(error),gate=gate,passed=math.isfinite(error)and abs(error)<=gate))
def rel(name,a,b,gate=3e-9):ck(name,(a-b)/max(abs(a),abs(b),1e-300),gate)
expected={'CONTRACT.md':'fbf5e19043e52d9872abd7807c2eaa54f6469390fd8b129565e779467039813c','MANIFEST.json':'71d256e0bfaed542498c396fad7885e744617850a57779186060d5aea64a13a5','CHECKPOINT.json':'4a50029a2a9e82798edd08e005fb793a85153b42aedb3c64888339623b46c7c7'}
for n,h in expected.items():assert sha(A/n)==h,n
M=json.loads((A/'MANIFEST.json').read_text());files=M.get('sha256',M.get('files'));assert all(sha(A/n)==h for n,h in files.items());pins=json.loads((A/'SOURCE_PIN.json').read_text());pins=pins.get('sha256',pins);assert all(sha(B/n)==h for n,h in pins.items());ck('allfrozenauthor/upstreampins')
ours=json.loads((P/'SOURCE_EXPECTATIONS.json').read_text());inventory=json.loads((A/'API_INVENTORY.json').read_text())
for label,key in [('leptons','PR51'),('baryons','PR52'),('weak','PR54')]:
 a=ours['interfaces'][label];z=inventory[key];assert a['sha256']==z['sha256']and a['functions']==z['functions'];ck('preregisteredactualAPI '+key)
 for name,fields in a['records'].items():assert fields==[f['name']for f in z['classes'][name]['fields']]
remote=json.loads((A/'PUBLISHED_SOURCE_READS.json').read_text())
for row in remote:assert row['result']['structuredContent']['content'].encode()==(B/inventory[row['label']]['path']).read_bytes()
ck('savedimmutablepublishedpublicsourcebytes')
# Attribute old executed records separately from current signature/source equality.
f=json.loads((B/'installed_free_matter_binding/forward_results.json').read_text())
for i,row in enumerate(f['states']):
 s,o=row['new'],row['old'];n,Y,T=s['nB'],s['Ye'],s['T']
 for sp,target in [('neutron',(1-Y)*n),('proton',Y*n)]:rel(f'forward{i}{sp}actualPARTICLE',s[sp]['particle_density'],target,1e-9)
 rel(f'forward{i}actualNETelectron',s['electron']['net_density'],n*Y,1e-9)
 for sp in ['neutron','proton','electron']:
  for key in ['chemical_potential','energy_density','pressure','entropy_density']:rel(f'forward{i}{sp}{key}',s[sp][key],o[sp][key])
ck('genuineforwardbeforederived',0 if all(c['passed']for c in checks)else 1)
w=json.loads((B/'installed_free_matter_binding/weak_fine.json').read_text());s=w['state'];n,T,Y=s['nB'],s['T'],s['Ye'];species=['neutron','proton','electron'];h={sp:s[sp]['chemical_potential']-T*s[sp]['dmu_dtemperature_fixed_net_density'if sp=='electron'else'dmu_dtemperature_fixed_particle_density']for sp in species};eY=n*(h['proton']-h['neutron']+h['electron']);en=(1-Y)*h['neutron']+Y*(h['proton']+h['electron']);cv=sum(s[sp]['cv_fixed_net_density'if sp=='electron'else'cv_fixed_particle_density']for sp in species)+s['photon_cv'];rel('fixedNET/PARTICLE eY',eY,s['e_Y']);rel('fixedTdensity en',en,s['e_n']);rel('fixeddensities Cv',cv,s['cv'])
for field,key in [('energy','energy_density'),('pressure','pressure'),('entropy','entropy_density')]:rel('sum '+field,s[field],sum(s[sp][key]for sp in species)+s['photon_'+field])
phe=math.pi**2*T**4/(15*197.3269804**3)
for field,val in [('energy',phe),('pressure',phe/3),('entropy',4*phe/(3*T)),('cv',4*phe/T)]:rel('LTEphoton '+field,s['photon_'+field],val)
mu={sp:s[sp]['chemical_potential']for sp in species};sum_n_muT=sum(s[sp]['particle_density'if sp!='electron'else'net_density']*s[sp]['dmu_dtemperature_fixed_net_density'if sp=='electron'else'dmu_dtemperature_fixed_particle_density']for sp in species);rel('PLUSsignenthalpy identity',s['energy']+s['pressure']-n*en,T*(s['entropy']+sum_n_muT),3e-9)
events={};bath_offsets=[]
for ch in ['EC','PC','NEUTRON_DECAY']:
 a=json.loads((B/f'spectral_band_total_closure/events/{ch}_empty.json').read_text());r=a['result'];assert a['complete']and a['bath']==r['bath']and a['radiation']['mode']=='empty';events[ch]=r
 for key,val in a['bath'].items():
  off=val-w['source_input']['bath'][key];ck(ch+'actual vs saved bath '+key,off,1e-11);bath_offsets.append(dict(channel=ch,key=key,absolute_offset=off))
 for k in[0,1]:assert r['forward'][k]['log']is not None and r['backward'][k]['value']==0. and r['backward'][k]['log']is None
 ck(ch+'positiveF_before_signed')
 st=dict(r['particle_stoichiometry']);ck(ch+'baryonchargeLe',abs(st['neutron']+st['proton'])+abs(st['proton']-st['electron_net'])+abs(st['electron_net']+st.get('nue',0)-st.get('antinue',0)))
R=events['PC']['net'][0]['value']+events['NEUTRON_DECAY']['net'][0]['value']-events['EC']['net'][0]['value'];Q=math.fsum(r['net'][1]['value']for r in events.values());rel('sourceR independent signed sum',R,w['source']['R']);rel('sourceQ independent signed sum',Q,w['source']['Q']);rel('Ye dot units',R/n,w['Ye_dot']);tdot=(((s['energy']+s['pressure'])/n-en)*w['ndot']-Q-eY*R/n)/cv;rel('densityenergy Tdot',tdot,w['T_dot'])
u=s['energy']/n;un=en/n-s['energy']/n**2;uY=eY/n;uT=cv/n;tdot2=((s['pressure']/n**2-un)*w['ndot']-Q/n-uY*R/n)/uT;rel('specificenergy equivalent chain',tdot,tdot2,3e-12)
zet=mu['proton']+mu['electron']-mu['neutron'];Ydot=R/n;edot=cv*tdot+en*w['ndot']+eY*Ydot;ni_dot={'neutron':(1-Y)*w['ndot']-n*Ydot,'proton':Y*w['ndot']+n*Ydot,'electron':Y*w['ndot']+n*Ydot};sigma_chain=(edot-math.fsum(mu[sp]*ni_dot[sp]for sp in species))/(T*n)-s['entropy']*w['ndot']/n**2;sigma_source=(-Q-zet*R)/(n*T);ck('entropy mechanwork cancels',(sigma_chain-sigma_source)/((abs(Q)+abs(zet*R))/(n*T)),3e-9)
rel('directmaterialloss no extra Q',math.fsum(r['net'][5]['value']for r in events.values()),Q)
# Proposed anti bound is derived, not evaluated or an existing API field.
ck('antibaryonsabsent in actual B API',0 if 'antiparticle_density'not in ours['interfaces']['baryons']['records']['ParticleState']else 1)
E=json.loads((A/'EVIDENCE.json').read_text());assert E['passed']==120 and not E['failed']and E['new_scientific_calls']==0
for key,val in [('R_fm3_s',R),('Q_MeV_fm3_s',Q),('Ye_dot_s',R/n),('T_dot_MeV_s',tdot)]:rel('author SAVED '+key,E['instantaneous'][key],val)
result=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,source_counts=dict(author=len(files),upstream=len(pins)),source_attribution='Previously executed saved forward/events, not fresh binder calls; current published source equality checked separately',derived=dict(R=R,Q=Q,Tdot=tdot,sigma_dot=sigma_source,pair_total_vs_net_ratio=s['electron']['particle_density']/s['electron']['net_density']),saved_bath_offsets=bath_offsets,proposed_anti_bound_computed=False,new_physical_calls=0,contract_implementation_qualified=False)
(P/'SAVED_REVIEW_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(result['passed'],'PASS',len(result['failed']),'FAIL',result['failed']);print(result['derived'])
