"""Independent integrated ledgers and local matched-EOS first-law diagnostic."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'nr_baryon_eos'))
import free_nucleons as eos
path=ROOT/'quantum_energy_moments/summary.json'
data=json.loads(path.read_text())
checks=[];diagnostics=[]
def add(name,error,tol):
 checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
for label,nn,np_ in [('dilute',7e-5,3e-5),('moderate',.002,.001)]:
 rows={r['flavor']:r['values'] for r in data['species'] if r['state']==label}
 n=nn+np_;Y=np_/n;T=2.;ncgs=n*1e39
 def matter(t,y):return eos.matter_state(t,n*(1-y),n*y)
 def u(st):return (st.baryon_energy_density+st.electron.energy_density)/n
 m=matter(T,Y)
 An=m.neutron.mu-T*m.neutron.dmu_dT_fixed_density
 Ap=m.proton.mu-T*m.proton.dmu_dT_fixed_density
 Ae=m.electron.chemical_potential-T*m.electron.dmu_dtemperature_fixed_net_density
 uY=Ap-An+Ae
 uT=(m.baryon_cv_fixed_densities+m.electron.cv_fixed_net_density)/n
 F=(rows['antinue']['emission_number_cm3_s']-rows['nue']['emission_number_cm3_s'])/ncgs
 q=sum(r['emission_energy_MeV_cm3_s'] for r in rows.values())/ncgs
 Tdot=(-q-uY*F)/uT
 h=1e-5*min(T/max(abs(Tdot),1e-100),Y/max(abs(F),1e-100),(1-Y)/max(abs(F),1e-100))
 fd=(u(matter(T+h*Tdot,Y+h*F))-u(matter(T-h*Tdot,Y-h*F)))/(2*h)
 add(label+' matched FD-EOS short-time empty-radiation energy closure',abs(fd+q)/max(abs(q),1.),2e-7)
 diagnostics.append(dict(state=label,ansatz='Instantaneous fixed-density n,p,e+- matter, no photons; rates frozen only for directional derivative',
                          dYe_dt_s_minus1=F,energy_loss_MeV_per_baryon_s=q,
                          dT_dt_MeV_s=Tdot,uY_MeV=uY,uT=uT,finite_difference_time_s=h))
 for flavor,r in rows.items():
  # Independent output-side flux ledger, not the positive-piece definition.
  gain=r['outgoing_left_number_cm2_s']+r['outgoing_right_number_cm2_s']-r['incoming_left_number_cm2_s']-r['incoming_right_number_cm2_s']
  scale=r['slab_emission_number_cm2_s']+r['slab_absorbed_boundary_number_cm2_s']
  add(label+' '+flavor+' independent four-boundary number ledger',abs(gain-r['radiation_number_gain_cm2_s'])/scale,1e-11)
  gainE=r['outgoing_left_energy_MeV_cm2_s']+r['outgoing_right_energy_MeV_cm2_s']-r['incoming_left_energy_MeV_cm2_s']-r['incoming_right_energy_MeV_cm2_s']
  scaleE=r['slab_emission_energy_MeV_cm2_s']+r['slab_absorbed_boundary_energy_MeV_cm2_s']
  add(label+' '+flavor+' independent four-boundary energy ledger',abs(gainE-r['radiation_energy_gain_MeV_cm2_s'])/scaleE,1e-11)
  add(label+' '+flavor+' MeV-erg energy sign/conversion',abs(r['bath_energy_source_erg_cm2_s']+gainE*1.602176634e-6)/max(abs(gainE*1.602176634e-6),1),1e-11)
  sign=1 if flavor=='nue' else -1
  add(label+' '+flavor+' matter electron-lepton sign',abs(r['bath_electron_lepton_source_cm2_s']+sign*gain)/scale,1e-11)
out=dict(scope='Independent serialized ledgers plus local EOS directional first law; no ODE or broad rate rerun',
 checks=checks,all_passed=all(c['passed'] for c in checks),instantaneous_diagnostics=diagnostics,
 summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
Path(__file__).with_name('energy_moment_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
