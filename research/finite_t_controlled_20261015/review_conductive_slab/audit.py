"""Independent SI, thermodynamic, ledger/figure provenance and source audit."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];H=Path(__file__).parent
sys.path.insert(0,str(R/'conductive_thermal_slab'));import slab
p=slab.Physics();checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
manifest=json.loads((R/'conductive_thermal_slab/MANIFEST.json').read_text())
for group,root in [('files',R/'conductive_thermal_slab'),('upstream',R)]:
 for name,digest in manifest[group].items():gate('frozen '+name,int(hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest),0)
# SI reconstruction from the conductivity and heat capacity in W/m/K,J/m³/K.
kb=1.380649e-23;hbar=6.582119569e-22;hc_m=197.3269804e-15
cv_SI=p.cv0*1e45*kb;k_SI=p.kappa0*kb/(hbar*hc_m);D=k_SI/cv_SI;t0=(1e-4)**2/D
sc=p.scales()
gate('independent diffusivity SI',D/sc['diffusivity_m2_per_s']-1,3e-15)
gate('independent time SI',t0/sc['time_unit_seconds']-1,3e-15)
s=p.thermo(p.T0);pf=197.3269804*(3*math.pi**2*p.ne)**(1/3);ef=math.hypot(pf,slab.L.ME)
nu=slab.L.frequency(pf,s);velocity=pf/ef*299792458.;tau=hbar/nu
gate('independent mfp SI',velocity*tau/sc['mfp_Fermi_m']-1,1e-8)
gate('independent Kn',velocity*tau/1e-4/sc['Knudsen_Fermi']-1,1e-8)
gate('fixed external ion density',s.ni_natural/(p.ne*197.3269804**3/p.Z)-1,2e-15)
# Direct thermodynamic derivatives at new held-out temperatures, not Cv0.
for theta in [.96,1.037]:
 h=2e-4;u1=p.thermo(p.T0*(theta+h)).gas['energy_density'];u0=p.thermo(p.T0*(theta-h)).gas['energy_density']
 s1=p.thermo(p.T0*(theta+h)).gas['entropy_density'];s0=p.thermo(p.T0*(theta-h)).gas['entropy_density']
 cv=p.thermo(p.T0*theta).gas['cv_fixed_density']
 gate('actual Cv heldout '+str(theta),(u1-u0)/(2*p.T0*h)/cv-1,2e-7)
 gate('fixed-density first law du=Tds '+str(theta),(u1-u0)/(p.T0*theta*(s1-s0))-1,2e-7)
 k=p.transport(p.T0*theta);K=np.array(k['K'])
 gate('open circuit thermoelectric covariance '+str(theta),p.kappa0*p.k(theta)/((K[2]-K[1]**2/K[0])/(p.T0*theta))-1,3e-15)
 gate('pair-free heldout domain '+str(theta),int(p.thermo(p.T0*theta).pair_log_ratio>math.log(1e-8)),0)
# Reconstruct original ledgers from actual saved endpoint temperatures.
d=json.loads((R/'conductive_thermal_slab/results.json').read_text())
for label in ['isolated','reservoir']:
 r=next(z['result']for z in d['rows']if z.get('label')==label and z['result']['controls']['steps']==64)
 E=float(np.mean([p.W(t)for t in r['theta']]));S=float(np.mean([p.entropy(t)for t in r['theta']]));last=r['history'][-1]
 gate('saved endpoint actual EOS energy '+label,E-last['material_energy'],3e-12)
 gate('saved endpoint actual EOS entropy '+label,S-last['entropy'],3e-12)
 gate('saved independent boundary energy '+label,E-r['initial_energy']-last['boundary_energy_in']+last['boundary_energy_out'],3e-10)
 total=S-r['initial_entropy']-last['boundary_entropy'];gate('saved physical entropy total '+label,min(0,total),1e-12)
 rows.append(dict(label=label,material_energy=E,material_entropy_change=S-r['initial_entropy'],boundary_entropy=last['boundary_entropy'],total_entropy=total,independent_face_discrepancy=total-last['face_entropy_production_integral']))
provenance=json.loads((R/'conductive_thermal_slab/figure_provenance.json').read_text())
gate('figure references exact final raw results',int(provenance['source_results_sha256']!=hashlib.sha256((R/'conductive_thermal_slab/results.json').read_bytes()).hexdigest()),0)
rows.append(dict(SI_heat_capacity_J_m3_K=cv_SI,SI_conductivity_W_m_K=k_SI,SI_diffusivity=D,SI_diffusion_time=t0,Fermi_velocity_m_s=velocity,Fermi_collision_time_s=tau))
(H/'audit.json').write_text(json.dumps(dict(checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
