from pathlib import Path
from dataclasses import asdict
import sys,json,math,time,signal
P=Path(__file__).resolve().parent;A=P.parent/'free_source_binder';sys.dont_write_bytecode=True;sys.path.insert(0,str(A/'staged/src'))
from nmbackend.reference import free_matter as F,weak_reference as W,_free_matter_sources as S
signal.alarm(89);start=time.monotonic();mode=sys.argv[1];checks=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
def save(name,data):
 (P/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
if mode=='direct':
 row=json.loads((P/'INDEPENDENT_FORWARD.json').read_text());T,n,Y=row['points'][0];z=row['oracles'][0]
 bath=W.FreeElectronBath(T,z['n']['mu'],z['p']['mu'],z['e']['mu']);rad=W.Radiation.empty();results=[W.channel_moments(ch,bath,rad,W.Controls.fine(ch))for ch in W.CHANNELS]
 saved=json.loads((P/'source_new_empty.json').read_text());orig=saved['source']['channel_results']
 for r,old in zip(results,orig):
  for direction in ['forward','backward']:
   for j,(q,o)in enumerate(zip(getattr(r,direction),old[direction])):
    if q.log is None:ck(r.channel+direction+str(j)+'empty inverse',0 if o['log']is None and q.value==0. else 1,0)
    else:ck(r.channel+direction+str(j)+'new absolute direct-root log',q.log-o['log'],3e-6)
 save('source_direct.json',dict(mode=mode,bath=asdict(bath),results=[asdict(r)for r in results],checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],seconds=time.monotonic()-start,independence='Fresh independent momentum chemical roots plus unmodified shared event/current APIs; independent assembly not different physical amplitude'))
else:
 if mode=='empty':T,n,Y,ph=1.7,8e-8,.27,'absent'
 elif mode=='LTE':T,n,Y,ph=2.6,1.8e-7,.41,'trapped_LTE'
 else:raise ValueError('Named prelocked source group')
 s=F.state_density(T,n,Y,photons=ph);eq=s.proton.chemical_potential+s.electron.chemical_potential-s.neutron.chemical_potential
 rad=W.Radiation.empty()if mode=='empty'else W.Radiation.fd(T,eq,-eq)
 source=F.source_moments(s,radiation=rad,controls=F.SourceControls.fine());rhs=F.instantaneous_rhs(s,source,baryon_density_dot_fm3_s=2.5e-8)
 rs=[r.net[0].value for r in source.channel_results];qs=[r.net[1].value for r in source.channel_results]
 grossR=sum(q.value for r in source.channel_results for q in[r.forward[0],r.backward[0]]);grossQ=sum(q.value for r in source.channel_results for q in[r.forward[1],r.backward[1]])
 R=rs[1]+rs[2]-rs[0];Q=math.fsum(qs);ck('R independently ordinary assembly',(source.proton_number_source.value-R)/grossR,2e-8);ck('Q independently ordinary assembly',(source.neutrino_energy_escape.value-Q)/grossQ,2e-8)
 material=0.
 for r in source.channel_results:
  v=[q.value for q in r.net]
  material+=v[4]-v[3]-v[2]if r.channel=='EC'else v[3]-v[4]-v[2]if r.channel=='PC'else v[3]+v[2]-v[4]
 ck('Independent species-energy closure',(material+Q)/grossQ,2e-8)
 ndot=2.5e-8;de=s.cv_fixed_nB_Ye*rhs.temperature_dot_mev_s+s.denergy_dnB_fixed_T_Ye*ndot+s.denergy_dYe_fixed_nB_T*rhs.Ye_dot_s
 # First-law check thermal/escape/work scales, never rest-energy denominator.
 work=(s.energy_density+s.pressure)*ndot/n;scale=abs(Q)+abs(work-s.denergy_dnB_fixed_T_Ye*ndot)+abs(s.denergy_dYe_fixed_nB_T*rhs.Ye_dot_s)
 ck('Density energy first law',(de-work+Q)/scale,2e-8)
 ck('Specific energy first law',((de/n-s.energy_density*ndot/n**2)-(s.pressure*ndot/n**2-Q/n))/(scale/n),2e-8)
 ck('Baryon net',(source.neutron_number_source.value+R)/grossR,2e-8);ck('Charge net',(source.electron_net_number_source.value-R)/grossR,2e-8)
 ck('Electron lepton net',(R+source.nue_number_source.value-source.antinue_number_source.value)/grossR,2e-8)
 if mode=='LTE':
  ck('LTE R gross',R/grossR,3e-6);ck('LTE Q gross',Q/grossQ,3e-6)
 # Entropy response independently reconstruct ds/dT=Cv/T, ds/dY=-n*(mu_pT+mu_eT-mu_nT), ds/dn=-sumfractions mu_iT.
 an=s.neutron.dmu_dtemperature_fixed_particle_density;ap=s.proton.dmu_dtemperature_fixed_particle_density;ae=s.electron.dmu_dtemperature_fixed_net_density
 ds=s.cv_fixed_nB_Ye/T*rhs.temperature_dot_mev_s-n*(ap+ae-an)*rhs.Ye_dot_s-((1-Y)*an+Y*(ap+ae))*ndot
 dsb=ds/n-s.entropy_density*ndot/n**2;entropy_scale=(grossQ+abs(eq)*grossR)/(n*T)
 ck('Material entropy chain',(dsb-rhs.material_entropy_per_baryon_dot_s)/entropy_scale,3e-6)
 save('source_new_'+('empty'if mode=='empty'else'LTE')+'.json',dict(mode=mode,state=asdict(s),source=S._payload(source),rhs=asdict(rhs),checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],seconds=time.monotonic()-start))
print(mode,'PASS',sum(x['passed']for x in checks),'FAIL',sum(not x['passed']for x in checks),'seconds',time.monotonic()-start);signal.alarm(0)
