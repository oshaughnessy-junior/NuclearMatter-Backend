"""Independent matched-EOS affinity and finite-M crossed-channel checks."""
from pathlib import Path
import sys,math,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'nr_baryon_eos'),str(ROOT/'crossed_opacity')]
import free_nucleons as eos
import crossed_decay as decay

ce,ee=[],[]
def add(bucket,name,error,tol):
    bucket.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))

T,n,Y=2.,1e-4,.3
def matter(t,y):return eos.matter_state(t,n*(1-y),n*y)
m=matter(T,Y)
def free_per_baryon(st):
    return (st.neutron.free_energy_density+st.proton.free_energy_density+
            st.electron.free_energy_density)/n
def energy_per_baryon(st):
    return (st.baryon_energy_density+st.electron.energy_density)/n
h=1e-4
fd=(free_per_baryon(matter(T,Y+h))-free_per_baryon(matter(T,Y-h)))/(2*h)
aff=m.optical('nue').mu_neutrino
add(ee,'neutral total-free-energy composition derivative equals weak affinity',
    abs(fd-aff)/max(1.,abs(aff)),1e-6)
An=m.neutron.mu-T*m.neutron.dmu_dT_fixed_density
Ap=m.proton.mu-T*m.proton.dmu_dT_fixed_density
Ae=m.electron.chemical_potential-T*m.electron.dmu_dtemperature_fixed_net_density
uY=Ap-An+Ae
fdE=(energy_per_baryon(matter(T,Y+h))-energy_per_baryon(matter(T,Y-h)))/(2*h)
add(ee,'FD baryon-plus-lepton composition energy derivative',abs(fdE-uY)/max(1.,abs(uY)),1e-6)
hT=2e-4*T
fdT=(energy_per_baryon(matter(T+hT,Y))-energy_per_baryon(matter(T-hT,Y)))/(2*hT)
Cv=(m.baryon_cv_fixed_densities+m.electron.cv_fixed_net_density)/n
add(ee,'total frozen-composition energy heat capacity',abs(fdT-Cv)/max(1.,abs(Cv)),1e-6)

ds=decay.DecayState(m.pair('neutron'),m.electron.chemical_potential)
add(ce,'shared EOS capture and decay antineutrino chemical parameter',
    abs(ds.mu_antineutrino-m.optical('antinue').mu_neutrino),1e-12)
for Enu in [.4,1.,2.]:
    lj,lc=decay.log_coefficients(ds,Enu,energy_order=32,angle_order=32)
    add(ce,f'finite-M FD-EOS crossed KMS E={Enu}',
        abs(lj-lc+(Enu-ds.mu_antineutrino)/T),2e-11)
    mix=decay.combined_antineutrino(ds,Enu,energy_order=48,angle_order=48)
    add(ce,f'finite-M FD-EOS combined KMS E={Enu}',
        abs(mix['log_j_s-1']-mix['log_chi_s-1']+(Enu-ds.mu_antineutrino)/T),2e-11)
    assert math.isfinite(lj) and math.isfinite(lc)

for filename,checks in [('crossed_probe_results.json',ce),('nr_eos_probe_results.json',ee)]:
    out=dict(scope='Independent selected shared-physics checks; no trajectory or full suite replay',
             checks=checks,all_passed=all(c['passed'] for c in checks))
    Path(__file__).with_name(filename).write_text(json.dumps(out,indent=2)+'\n')
    print(filename,json.dumps(out))
