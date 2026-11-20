"""Independent trajectory/ledger comparison; separates source and time errors."""
from common import *
checks=[];diagnostics={}
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
review=json.loads((ROOT/'review_transparent_capture_balance/CHECKPOINT.json').read_text())
refpath='review_hot_electron_capture/probe.py';ck('accepted alternate EC reference source SHA',int(hashlib.sha256((ROOT/refpath).read_bytes()).hexdigest()!=review['frozen_dependencies'][refpath]),0)
oldpath='transparent_capture_balance/results.json';ck('accepted initial source record SHA',int(hashlib.sha256((ROOT/oldpath).read_bytes()).hexdigest()!=review['frozen_dependencies'][oldpath]),0)
old=json.loads((ROOT/oldpath).read_text())['rows'][0]['refined']['channels'];fresh=json.loads((P/'forward.json').read_text())['rows'][0]['packet']['sources']['channels']
for c in old:
 for k,v in old[c].items():ck('fresh initial versus accepted source '+c+' '+k,fresh[c][k]/v-1,3e-6)
a=json.loads((P/'rk4_1.json').read_text());b=json.loads((P/'rk4_2.json').read_text());author=json.loads((A/'final_replay.json').read_text())['trajectory'];initial=author['initial'];final=author['final'];duration=author['duration']
ck('RK4 selected time escape refinement',a['escaped_energy']/b['escaped_energy']-1,1e-3)
ck('RK4 selected time COMPOSITION CHANGE refinement',(a['final']['Ye']-b['final']['Ye'])/(b['final']['Ye']-.3),1e-3)
ck('RK4 versus Heun4 escape',author['escaped_energy']/b['escaped_energy']-1,1e-3)
ck('RK4 versus Heun4 COMPOSITION CHANGE',(final['Ye']-b['final']['Ye'])/(b['final']['Ye']-.3),1e-3)
ck('RK4 versus Heun4 temperature CHANGE',(final['T']-b['final']['T'])/(b['final']['T']-2.),1e-3)
# Independently reassemble literal Heun weights from stored stage primitives, not stored totals.
rate={c:0. for c in ['beta','PC','EC']};q={c:0. for c in rate};dt=duration/author['steps']
for record in author['records']:
 for z in [record['source'],record['second_source']]:
  for c in rate:rate[c]+=dt/2*z['channels'][c]['number'];q[c]+=dt/2*z['channels'][c]['neutrino_energy']
E=sum(q.values());nu=rate['EC'];anti=rate['beta']+rate['PC'];gross=nu+anti
for c in rate:
 ck('Heun literal-stage number '+c,rate[c]/author['integrated'][c]['number']-1,3e-14)
 ck('Heun literal-stage energy '+c,q[c]/author['integrated'][c]['energy']-1,3e-14)
ck('Heun restinclusive exchanged-energy ledger',(final['energy_density']+E-initial['energy_density'])/E,2e-8)
ck('Heun independent integrated Le',(NB*(final['Ye']-initial['Ye'])+nu-anti)/gross,2e-13)
weakminus=rate['beta']-rate['EC'];weakplus=-rate['PC'];emminus=final['electron_particle_density']-initial['electron_particle_density']-weakminus;emplus=final['positron_density']-initial['positron_density']-weakplus
ck('Heun independently inferred EMpair equality',(emminus-emplus)/gross,2e-12)
ck('Heun EMpair count from electrons',emminus/author['pair_EMnumber_change']['from_electron']-1,3e-12)
ck('Heun photon plus material loss once',(initial['photon']['energy_density']-final['photon']['energy_density']+sum(initial['material'][k]['energy_density']-final['material'][k]['energy_density']for k in ['neutron','proton','electron'])-E)/E,2e-8)
ck('Heun no external heat or photon escape',int(author['thermostat_energy']!=0 or author['photon_energy_escape']!=0),0)
# Same source, independently finite-difference EOS heat capacity with/without photon reservoir.
src=json.loads((P/'forward.json').read_text())['rows'][0]['packet']['sources'];h=1e-4;dy=1e-5
uT=(eos(2+h,.3)['u']-eos(2-h,.3)['u'])/(2*h);uY=(eos(2,.3+dy)['u']-eos(2,.3-dy)['u'])/(2*dy);cvg=4*math.pi**2*2**3/(15*HC**3*NB);tdot=(src['u_dot']-uY*src['Ye_dot'])/uT;tdotoff=(src['u_dot']-uY*src['Ye_dot'])/(uT-cvg)
ck('independent initial photonON Tdot',tdot/(-1.89788065940)-1,3e-7);ck('independent initial photonOFF Tdot',tdotoff/(-2.95345818074)-1,3e-7)
diagnostics=dict(RK4_1=dict(T=a['final']['T'],Ye=a['final']['Ye'],E=a['escaped_energy']),RK4_2=dict(T=b['final']['T'],Ye=b['final']['Ye'],E=b['escaped_energy']),photonON_Tdot=tdot,photonOFF_Tdot=tdotoff,Heun_EMpair_change=emminus,Heun_photon_loss=initial['photon']['energy_density']-final['photon']['energy_density'],RK4_EMpair_change=b['pair_EM_from_electron'],RK4_photon_loss=b['initial']['photon_energy']-b['final']['photon_energy'])
ck('all frozen bytes still unchanged',int(json.loads((P/'before.json').read_text())!=frozen()),0)
(P/'comparison.json').write_text(json.dumps(dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],diagnostics=diagnostics),indent=2)+'\n');print(len(checks),[c for c in checks if not c['passed']]);print(diagnostics)
