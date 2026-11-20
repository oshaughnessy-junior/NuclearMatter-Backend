"""New heldout cold-shell and finite-T p_perp/spinor review witness."""
from pathlib import Path
import sys,json,time,math,hashlib,atexit
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
for p in ['gm1_beta_thresholds','gm1_differential_response','relativistic_free_eos','relativistic_delta_oracle']:sys.path.insert(0,str(ROOT/p))
import cold_beta as B,adapter as A,free_baryons as F,oracle as O
start=time.monotonic();checks=[];rows=[];phase='start';files=[ROOT/'gm1_threshold_thermal_probes'/f for f in ['probe.py','results.json','heldout_results.json','boltzmann_results.json','initial_heldout_scope_failure.py']]+[ROOT/'gm1_differential_response/adapter.py',ROOT/'relativistic_delta_oracle/oracle.py'];hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in files}
def save():
 end={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in files}
 (HERE/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks)and hashes==end,phase=phase,checks=checks,rows=rows,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256=end),indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item())+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def density(T,mu,M,pair=False):
 energies=sorted(set([M,max(M,abs(mu))+80*T]+[s+k*T for s in [mu,-mu]for k in[-12,-4,0,4,12]if M<s+k*T<max(M,abs(mu))+80*T]));cuts=[math.sqrt((E-M)*(E+M))for E in energies]
 def fun(p):
  E=math.hypot(M,p);return p*p*(expit((mu-E)/T)-(expit((-mu-E)/T)if pair else 0))/(math.pi**2*B.H**3)
 return sum(quad(fun,l,h,epsabs=1e-17,epsrel=2e-11)[0]for l,h in zip(cuts[:-1],cuts[1:]))

def independent(s,Enu,El,c,ml,mul,order,tail):
 # No q-axis initial-energy integration: solve original longitudinal delta at fixed transverse momentum.
 M=s.dirac_mass;pl=math.sqrt((El-ml)*(El+ml));q=math.sqrt((Enu-pl)**2+2*Enu*pl*(1-c));omega=Enu-El;w=omega-s.potential_b+s.potential_a
 if q<=abs(w):raise ValueError('Selected spacelike kinetic particle-hole oracle only')
 kx=Enu*pl*math.sqrt(1-c*c)/q;kz=Enu*(Enu-pl*c)/q;k=np.array([Enu,kx,0,kz]);l=np.array([El,kx,0,kz-q]);qstar=np.array([w,0,0,q]);t=omega*omega-q*q;mb=s.weak_bare_mass
 den=(1-t/(4*mb*mb))*(1-t/840.**2)**2;v=(1-4.706*t/(4*mb*mb))/den;a=1.2723/(1-t/1000.**2)**2;f=3.706/den;gp=2*mb*mb*a/(139.57**2-t)
 V=np.array([v*O.G[u]-a*O.G[u]@O.G5-f/(4*mb)*sum((O.G[u]@O.G[j]-O.G[j]@O.G[u])*O.metric[j]*qstar[j]for j in range(4))-gp/mb*qstar[u]*O.G5 for u in range(4)])
 uk=O.spinors(k,0);ul=O.spinors(l,ml);Lc=[]
 for uu in ul:
  for kk in uk:Lc.append([O.bar(uu)@O.G[i]@(O.I-O.G5)@kk for i in range(4)])
 LL=np.asarray(Lc).T;L=LL@LL.conj().T
 nua=s.total_mu_a-s.potential_a;nub=s.total_mu_b-s.potential_b
 def shell(r):
  def delta(z):return math.sqrt(M*M+r*r+z*z)+w-math.sqrt(M*M+r*r+(z+q)**2)
  z=brentq(delta,-1e7,1e7,xtol=2e-10,rtol=2e-14);Ea=math.sqrt(M*M+r*r+z*z);Eb=math.sqrt(M*M+r*r+(z+q)**2);J=abs(z/Ea-(z+q)/Eb);return z,Ea,Eb,J,abs(Ea+w-Eb)
 z,E0,Eb,J,res=shell(0.);xmax=tail # transverse energy measured in T; convergence tested explicitly.
 cuts=sorted(set([0.,1.,3.,8.,16.,32.,xmax]));x,g=leggauss(order);W=np.zeros((2,4,4),complex);maxdelta=0.
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  for xi,wi in zip((lo+hi)/2+(hi-lo)*x/2,g*(hi-lo)/2):
   r=math.sqrt(2*M*s.T*xi);z,Ea,Eb,J,res=shell(r);maxdelta=max(maxdelta,res);occ=np.array([expit((nua-Ea)/s.T)*expit((Eb-nub)/s.T),expit((nub-Eb)/s.T)*expit((Ea-nua)/s.T)])
   for phi in 2*math.pi*np.arange(8)/8:
    pa=np.array([Ea,r*math.cos(phi),r*math.sin(phi),z]);pb=np.array([Eb,pa[1],pa[2],z+q]);ua=O.spinors(pa,M);ub=O.spinors(pb,M);curr=[]
    for b in ub:
     for a in ua:curr.append([O.bar(b)@vv@a for vv in V])
    Jv=np.asarray(curr).T;H=Jv@Jv.conj().T;W+=occ[:,None,None]*H[None,:,:]*M*s.T*wi*(2*math.pi/8)/(16*math.pi**2*Ea*Eb*J)
 pref=O.GF**2*O.VUD**2/(32*math.pi**2)*pl/Enu/O.HBAR;con=np.real(np.sum(np.asarray(O.metric)[:,None]*np.asarray(O.metric)[None,:]*L[None,:,:]*W,axis=(1,2)));coef=pref*con*np.array([expit((El-mul)/s.T),expit((mul-El)/s.T)])
 return dict(chi=float(coef[0]),j=float(coef[1]),W=W,max_delta=maxdelta,Emin_original_delta=E0,q=q,w=w,physical_t=t,kinetic_t=w*w-q*q,controls=dict(order=order,tail_transverse_T=tail))
# New T/angle, fixed cold beta fractions at density .97 of zero-neutrino triangle threshold.
raw=json.loads((ROOT/'gm1_threshold_thermal_probes/results.json').read_text());ncrit=raw['cold_onset']['nB'];cold=B.state(.97*ncrit);T=.21;Enu=.3;El=cold.mu_lepton;c=-.67;ml=B.ME
N=cold.nuclear;M=N.dirac_mass_mev;Un=N.omega_mev-.5*N.rho_mev;Up=N.omega_mev+.5*N.rho_mev
neutron=F.state_density(T,cold.nB*(1-cold.Yp),M);proton=F.state_density(T,cold.nB*cold.Yp,M);electron=A.leptons.state_density(T,cold.nB*cold.Ye,ml);muon=A.leptons.state_density(T,cold.nB*cold.Ymu,B.MMU);frozen=A.OrderedState(T,M,Un,Up,neutron.chemical_potential+Un,proton.chemical_potential+Up,939.);warm=A.matter_state(T,cold.nB,cold.Ye,cold.Ymu)
for label,s,e,mu in [('frozen',frozen,electron,muon),('warm',warm.ordered(),warm.electron,warm.muon)]:
 phase=label+' forward';targets=[cold.nB*(1-cold.Yp),cold.nB*cold.Yp,cold.nB*cold.Ye,cold.nB*cold.Ymu]
 for mass,chem,target,pair in [(s.dirac_mass,s.total_mu_a-s.potential_a,targets[0],False),(s.dirac_mass,s.total_mu_b-s.potential_b,targets[1],False),(ml,e.chemical_potential,targets[2],True),(B.MMU,mu.chemical_potential,targets[3],True)]:
  gate(label+' forward momentum density',density(T,chem,mass,pair)/target-1,3e-9)
  root=brentq(lambda v:density(T,v,mass,pair)/target-1,chem-5*T,chem+5*T,xtol=5e-11,rtol=2e-14);gate(label+' independent chemical root',root-chem,3e-8)
 phase=label+' response';refs=[independent(s,Enu,El,c,ml,e.chemical_potential,order,tail)for order,tail in[(20,60),(32,100)]];actual=A.differential(s,Enu,El,c,ml,e.chemical_potential,energy_order=48,phi_order=8,tail=90)
 for name in['chi','j']:
  gate(label+' independent original-delta '+name,refs[-1][name]/actual[name]['value']-1,3e-7);gate(label+' transverse refinement '+name,refs[0][name]/refs[-1][name]-1,3e-7)
 for j,name in enumerate(['absorption','emission']):gate(label+' full W '+name,np.linalg.norm(refs[-1]['W'][j]-actual['response'][name]['tensor'])/np.linalg.norm(refs[-1]['W'][j]),3e-7)
 gate(label+' original delta residual',refs[-1]['max_delta'],3e-8)
 # Exact cold occupations at this SAME Hamiltonian and same finite physical kinematics.
 kn=B.H*(3*math.pi**2*targets[0])**(1/3);kp=B.H*(3*math.pi**2*targets[1])**(1/3);nuNF=math.hypot(s.dirac_mass,kn);nuPF=math.hypot(s.dirac_mass,kp);w=refs[-1]['w'];lo=max(refs[-1]['Emin_original_delta'],nuPF-w);gap=nuNF-lo;gate(label+' cold occupation window is strictly empty',float(gap>=0),0.);gate(label+' thermal positive absorption',float(refs[-1]['chi']<=0),0.)
 rows.append(dict(closure=label,T=T,nB=cold.nB,Ye=cold.Ye,Ymu=cold.Ymu,Enu=Enu,El=El,cosine=c,weak_bare_mass=s.weak_bare_mass,affinity=e.chemical_potential+s.total_mu_b-s.total_mu_a,cold_same_Hamiltonian_gap=gap,cold_nu_n=nuNF,cold_nu_p=nuPF,independent=refs,actual_chi=actual['chi'],actual_j=actual['j']));save()
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
