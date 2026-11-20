"""Held-out irregular-grid neutral-current entropy and reciprocity probes."""
from pathlib import Path
import sys,json,math
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'neutral_current'))
import free_nc as nc

checks=[]
def add(name,error,tol):
 checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
s=nc.state_density(1.3,.0013,offset=17.)
E=np.array([3.5,5.,7.5,11.,18.]);w=np.array([.3,1.3,.8,2.1,4.])
couplings=nc.Couplings(.2,.7)
for mu in [-20.,7.,30.]:
 f=expit((mu-E)/s.T)
 r=nc.discrete_collision(s,E,w,f,couplings,angle_order=40)
 g,A=r['density_weights'],r['transition_rates']
 flux=g[:,None]*A*f[:,None]*(1-f[None,:])
 gross=flux.sum()
 add(f'held-out irregular-grid arbitrary-mu stationary {mu}',max(abs(r['number_source']))/gross,2e-12)

f=np.array([.91,.07,.63,.21,.005])
r=nc.discrete_collision(s,E,w,f,couplings,angle_order=40)
g,A=r['density_weights'],r['transition_rates']
flux=g[:,None]*A*f[:,None]*(1-f[None,:]);gross=flux.sum()
add('held-out irregular-grid number conservation',abs(r['net_number_source'])/gross,1e-13)
macro=-np.dot(r['number_source'],np.log(f)-np.log1p(-f)+E/s.T)
pairwise=0.
paired_dN=np.zeros(len(E))
for i in range(len(E)):
 for j in range(i):
  F,B=flux[i,j],flux[j,i]
  pairwise+=(F-B)*math.log(F/B)
  paired_dN[i]+=B-F;paired_dN[j]+=F-B
add('pairwise (F-B)log(F/B) versus radiation-plus-bath entropy',abs(macro/pairwise-1),2e-12)
paired_macro=-np.dot(paired_dN,np.log(f)-np.log1p(-f)+E/s.T)
add('paired off-diagonal accumulation entropy identity',abs(paired_macro/pairwise-1),2e-12)
checks.append(dict(name='held-out entropy production nonnegative',value=float(macro),passed=bool(macro>=0)))
bath=sum((E[i]-E[j])*flux[i,j] for i in range(len(E)) for j in range(len(E)))
add('held-out signed bath-versus-radiation energy',abs(bath+r['net_neutrino_energy_source'])/(gross*max(E)),1e-13)

pair=nc.pair_from_species(s)
for Ein,Eout,c in [(6.3,8.7,-.42),(13.5,9.2,.73)]:
 lr=nc.log_kernel(pair,Eout,Ein,c,nc.PROTON)
 lf=nc.log_kernel(pair,Ein,Eout,c,nc.PROTON)
 add(f'held-out proton DOS-ratio KMS {Ein,Eout,c}',
     abs(lr-lf-2*math.log(Ein/Eout)+(Ein-Eout)/s.T),2e-12)
out=dict(scope='Independent held-out algebraic/grid physics checks; no continuum-rate claim',
 checks=checks,all_passed=all(c['passed'] for c in checks),
 entropy_production_fm_minus3_s_minus1=float(macro),bath_energy_source_MeV_fm_minus3_s_minus1=float(bath),
 paired_entropy_production=float(paired_macro),pairwise_entropy_production=float(pairwise),
 max_pairwise_vs_original_dN_difference=float(max(abs(paired_dN-r['number_source']))),
 gross_transition_flux=float(gross),diagonal_transition_flux=float(np.trace(flux)))
Path(__file__).with_name('nc_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
