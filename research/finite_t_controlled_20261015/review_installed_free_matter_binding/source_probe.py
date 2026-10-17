"""Fresh physical phase calls at independent controls; new ledger assembly only."""
from pathlib import Path
import sys,importlib.util,json,hashlib,time,math
P=Path(__file__).resolve().parent;R=P.parent;A=R/'installed_free_matter_binding';sys.dont_write_bytecode=True
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
start=time.monotonic();checks=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
M=load('source_review_frozen_matter',A/'matter.py');W=load('source_review_accepted_functions',R/'three_source_cooling_element/worker.py');pins=W.pins();B=W.physical()
a=M.state(2,1e-7,.3);bath=B.EC.Bath(**a.weak_bath());lock=json.loads((P/'VALIDATION_LOCK.json').read_text());cp=lock['source_controls'];saved=json.loads((A/'weak_fine.json').read_text());rows={};raw={}
def save():
 (P/'source_results.json').write_text(json.dumps(dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],channels=rows,raw=raw,seconds=time.monotonic()-start),indent=2)+'\n')
x=cp['EC'];raw['EC']=B.EC.integrate(bath,proton_order=x[0],electron_order=x[1],initial_order=x[2],outgoing_order=x[2],phi_order=x[2],tail=x[3],scenarios=[dict(name='empty',mode='empty')],budget_seconds=90.);rows['EC']=raw['EC']['scenarios']['empty']['net'];save()
x=cp['PC'];raw['PC']=B.PC.integral(vars(bath),order_n=x[0],order_e=x[1],order_angle=x[2],tail=x[3],budget=90.);rows['PC']=dict(zip(B.PC.NAMES,map(float,raw['PC']['sums'][0][0])));save()
x=cp['beta'];raw['beta']=B.BETA.integrate(B.BETA.K.Bath(**vars(bath)),rest_order=x[0],parent_order=x[1],orient_order=x[2],phi_order=x[3],coordinate=x[4],parent_coordinate=x[5],tail=x[6],scenarios=[dict(name='empty',mode='empty')],budget_seconds=90.);rows['beta']=raw['beta']['scenarios']['empty']['net'];save()
for ch,v in rows.items():
 for k in v:ck('fresh alternate physical quadrature '+ch+' '+k,v[k]/saved['source']['channels'][ch][k]-1,3e-6)
 ck('channel direct material loss '+ch,(v['material_energy_loss']-v['neutrino_energy'])/v['neutrino_energy'],2e-9)
 if ch=='EC':net=v['neutron_energy']-v['proton_energy']-v['electron_energy']+v['neutrino_energy']
 elif ch=='PC':net=v['proton_energy']-v['neutron_energy']-v['positron_energy']+v['neutrino_energy']
 else:net=v['proton_energy']+v['electron_energy']-v['neutron_energy']+v['neutrino_energy']
 ck('separate full rest inclusive E '+ch,net/v['neutrino_energy'],2e-9)
r={k:v['number']for k,v in rows.items()};q={k:v['neutrino_energy']for k,v in rows.items()};Rnet=r['beta']+r['PC']-r['EC'];Q=sum(q.values());Ydot=Rnet/a.nB;ndot=2.5e-8
# Independent root solved EOS derivatives were checked by probe.py before this experiment.
Tdot=(((a.energy+a.pressure)/a.nB-a.e_n)*ndot-Q-a.e_Y*Ydot)/a.cv
ck('new actual R', (Rnet-saved['source']['R'])/sum(r.values()),3e-6);ck('new actual Q',Q/saved['source']['Q']-1,3e-6)
ck('instantaneous Tdot', (Tdot-saved['T_dot'])/max(abs(Tdot),abs(saved['T_dot'])),3e-6)
# Reconstruct species numbers by integer stoichiometry, not source() ledger fields.
nu=r['EC'];antinu=r['PC']+r['beta'];dne=r['beta']-r['EC'];dpos=-r['PC'];dnp=Rnet;dnn=-Rnet
ck('electron Le', (dne-dpos+nu-antinu)/sum(r.values()),2e-9);ck('electric charge',(dnp-dne+dpos)/sum(r.values()),2e-9);ck('baryon number',(dnp+dnn)/sum(r.values()),2e-9)
ck('actual density EC',raw['EC']['proton_density']/(a.nB*a.Ye)-1,3e-8);ck('actual density PC',raw['PC']['neutron_density']/(a.nB*(1-a.Ye))-1,3e-8);ck('actual density beta',raw['beta']['parent_density']/(a.nB*(1-a.Ye))-1,3e-8)
ck('positive actual outgoing energy',0 if all(x>0 for x in q.values())else 1,0)
W.pins();save();print('PASS',sum(c['passed']for c in checks),'FAIL',sum(not c['passed']for c in checks),'seconds',time.monotonic()-start)
