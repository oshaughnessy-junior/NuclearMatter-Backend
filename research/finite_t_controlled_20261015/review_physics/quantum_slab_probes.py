"""Independent signed-source and collapsed-panel estimates for slab review."""
from pathlib import Path
import sys,math,json
import numpy as np
from scipy.integrate import quad
from scipy.special import log_expit,logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'quantum_slab'))
import fixed_bath as slab
import guarded_capture as guard
from free_density_response import log_response
from allowed_capture import nodes,angular_factor,K,ME

checks=[]
def add(name,error,tol):
 checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
# Synthetic absorbing bath checks the sign of the material exchange and
# independently integrates the actual j(1-f)-chi*f source over depth/angle.
coeff=dict(energy_MeV=3.,feq=.2,j_s=2.,chi_s=8.,rate_s=10.,
           relaxation_cm_inv=10/slab.LIGHT,true_absorption_cm_inv=8/slab.LIGHT)
for flavor in ['nue','antinue']:
 coeff['flavor']=flavor
 for tau in [.03,1.,20.]:
  L=tau/coeff['relaxation_cm_inv']; left,right=.8,.6
  computed=slab.fluxes(coeff,L,left,right)
  def source_at_t(t):
   def f(mu):
    ff=slab.ray_occupation(t,mu,tau,.2,left,right)
    return (2*(1-ff)-8*ff)/10
   return .5*(quad(f,-1,0,epsabs=1e-11)[0]+quad(f,0,1,epsabs=1e-11)[0])
  # Break points explicitly resolve both absorbing surfaces.
  edges=sorted(set([0.,tau]+[x for x in [.01,.1,1.,tau-.01,tau-.1,tau-1.] if 0<x<tau]))
  volume=sum(quad(source_at_t,a,b,epsabs=2e-10,epsrel=2e-9)[0]
             for a,b in zip(edges[:-1],edges[1:]))
  A=slab.LIGHT*slab.DOS*3**2*1e39
  add(f'direct signed collision-volume source {flavor} tau={tau}',
      abs(volume*A/computed['radiation_number_gain']-1),2e-7)
  assert computed['bath_energy_source_MeV']>0
  assert computed['bath_electron_lepton_source']*(1 if flavor=='nue' else -1)>0

# Estimate raw-E contribution of the ulp-sized panels that cannot be represented
# in x; this is a selected smooth-integrand estimate, not a universal bound.
m=slab.matter_state(2.,.002,.001); state=m.optical('nue'); E=60.
lj,lc,discard=guard.log_coefficients(state,E,energy_order=72,angle_order=72,tail=50.)
cs,cws=nodes(72); termsj=[];termsc=[]
for d in discard:
 c=d['cosine'];cw=cws[np.argmin(abs(cs-c))];Ee=(d['lo']+d['hi'])/2
 pe=math.sqrt(Ee*Ee-ME*ME);q=math.sqrt(E*E+pe*pe-2*E*pe*c)
 common=math.log(K*cw*d['width']*Ee*pe*angular_factor(Ee,c))
 termsc.append(common+log_expit((Ee-state.mu_lepton)/2)+log_response(state.baryons,q,E-Ee))
 termsj.append(common+log_expit((state.mu_lepton-Ee)/2)+log_response(state.baryons.reverse(),q,Ee-E))
ratios=[math.exp(logsumexp(termsj)-lj),math.exp(logsumexp(termsc)-lc)]
add('estimated discarded-panel j/chi fractions at recorded failure',max(ratios),1e-12)
out=dict(scope='Selected independent source integrals and discarded-panel estimate',
         checks=checks,all_passed=all(c['passed'] for c in checks),
         collapsed_panels=len(discard),estimated_omitted_j_chi_fractions=ratios)
Path(__file__).with_name('quantum_slab_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
