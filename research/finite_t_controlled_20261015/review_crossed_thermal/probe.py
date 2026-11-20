"""Independent transverse-radius/two-root original-delta crossed check."""
from pathlib import Path
import sys,math,json,time,hashlib
import numpy as np
from scipy.special import log_expit,logsumexp
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'relativistic_crossed_thermal'));import kernel as K
sys.path.insert(0,str(ROOT/'relativistic_delta_oracle'));import oracle as O
checks=[]
def ck(name,e,t):checks.append(dict(name=name,error=abs(float(e)),limit=t,passed=bool(abs(e)<=t)))

def vertex(q,M):
 s=q[0]*q[0]-q[1:]@q[1:];den=(1-s/(4*M*M))*(1-s/840.**2)**2
 v=(1-4.706*s/(4*M*M))/den;a=1.2723/(1-s/1000.**2)**2;f=3.706/den;g=2*M*M*a/(139.57**2-s)
 return np.array([v*O.G[u]-a*O.G[u]@O.G5-f/(4*M)*sum((O.G[u]@O.G[j]-O.G[j]@O.G[u])*O.MET[j]*q[j]for j in range(4))-g/M*q[u]*O.G5 for u in range(4)])

def independent(bath,Enu,Ee,c,order=20):
 pe=math.sqrt((Ee-K.ME)*(Ee+K.ME));om=Enu+Ee;q=math.sqrt(Enu*Enu+pe*pe+2*Enu*pe*c);s=om*om-q*q;rt=math.sqrt(s)
 assert K.ME**2<s<(K.MN-K.MP)**2
 Estar=((K.MN-K.MP)*(K.MN+K.MP)-s)/(2*rt);R=math.sqrt((Estar-K.MP)*(Estar+K.MP));gamma=om/rt;velocity=q/om
 kz=(q*q+Enu*Enu-pe*pe)/(2*q);kx=math.sqrt(max(0.,Enu*Enu-kz*kz));k=np.array([Enu,kx,0.,kz]);l=np.array([Ee,-kx,0.,q-kz]);Q=np.array([om,0.,0.,q]);J=vertex(Q,(K.MN+K.MP)/2)
 ue=O.spinors(l,K.ME);va=O.spinors(k,0,True);left=O.G@(O.I-O.G5)
 L=np.array([(va.conj().T@O.G[0]@left[u]@ue).reshape(-1)for u in range(4)]);Ldec=np.array([(ue.conj().T@O.G[0]@left[u]@va).reshape(-1)for u in range(4)])
 # r=R sin(theta), BOTH longitudinal roots. Original delta Jacobian is
 # |v_n,z-v_p,z|; dr vanishes at root coalescence, resolving its endpoint.
 umax=R*R/(2*K.MP);cuts=[0.,math.pi/2]
 for multiple in(.3,1.,3.,8.,20.,50.,90.):
  u=multiple*bath.T
  if u<umax:cuts.append(math.asin(math.sqrt(u/umax)))
 cuts=sorted(cuts);x,w=leggauss(order);logs=dict(j=[],chi=[]);maxshell=0.;reciprocity=0.;rootcount=0
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  for xx,ww in zip(x,w):
   theta=lo+(xx+1)*(hi-lo)/2;rad=R*math.sin(theta);drweight=R*R*math.sin(theta)*math.cos(theta)*ww*(hi-lo)/2
   for sign in(-1,1):
    zstar=sign*R*math.cos(theta);z=gamma*(zstar+velocity*Estar);Ep=gamma*(Estar+velocity*zstar);En=Ep+om;jac=abs((z+q)/En-z/Ep);rootcount+=1
    occ_p=float(log_expit((bath.mu_p-Ep)/bath.T));vac_p=float(log_expit((Ep-bath.mu_p)/bath.T));occ_n=float(log_expit((bath.mu_n-En)/bath.T));vac_n=float(log_expit((En-bath.mu_n)/bath.T))
    for phi in 2*math.pi*np.arange(8)/8:
     pp=np.array([Ep,rad*math.cos(phi),rad*math.sin(phi),z]);pn=pp+Q;up=O.spinors(pp,K.MP);un=O.spinors(pn,K.MN)
     had=np.array([(un.conj().T@O.G[0]@J[u]@up).reshape(-1)for u in range(4)])
     amp=np.einsum('ui,u,uj->ij',L,O.MET,had);a=float(np.sum(abs(amp)**2));assert a>0
     # Decay uses reciprocal current evaluated at negative Q, not the
     # unchanged antineutrino positron-capture trace.
     decay_vertex=vertex(-Q,(K.MN+K.MP)/2)
     hdec=np.array([(up.conj().T@O.G[0]@decay_vertex[u]@un).reshape(-1)for u in range(4)])
     adec=float(np.sum(abs(np.einsum('ui,u,uj->ij',Ldec,O.MET,hdec))**2));reciprocity=max(reciprocity,abs(adec/a-1))
     pref=O.GF**2*O.VUD**2*pe/(256*math.pi**3*Enu*O.HBAR)
     logweight=math.log(pref*drweight/(8*Ep*En*jac))+math.log(a)
     logs['chi'].append(logweight+occ_p+vac_n+float(log_expit((bath.mu_e-Ee)/bath.T)))
     logs['j'].append(logweight+occ_n+vac_p+float(log_expit((Ee-bath.mu_e)/bath.T)))
     maxshell=max(maxshell,abs(Ep*Ep-np.dot(pp[1:],pp[1:])-K.MP**2),abs(En*En-np.dot(pn[1:],pn[1:])-K.MN**2),abs(En-Ep-om))
 return dict(j=float(logsumexp(logs['j'])),chi=float(logsumexp(logs['chi'])),max_shell_absolute=maxshell,max_reciprocity=reciprocity,root_evaluations=rootcount,invariant=s,Estar=Estar,pstar=R,order=order,prefactor='GF²Vud² pe/(256pi³ Enu hbar), full spin sums and phi average; no target or neutrino spin average')

start=time.monotonic();bath,material=K.matched_npe(.045,3e-12,.37);rows=[]
for Enu,Ee,c in[(.27,1.023,.15),(.43,.863,-.4),(.82,.526,.85)]:
 runs=[independent(bath,Enu,Ee,c,order=n)for n in(12,20)]
 actual=[K.differential(bath,Enu,Ee,c,energy_order=n,phi_order=8,tail=80.)for n in(24,40)]
 for name in('j','chi'):
  ck('independent_two_delta_roots_'+name,math.expm1(actual[-1][name]['log']-runs[-1][name]),3e-7)
  ck('independent_radial_refinement_'+name,math.expm1(runs[0][name]-runs[-1][name]),3e-7)
  ck('kernel_energy_refinement_'+name,math.expm1(actual[0][name]['log']-actual[-1][name]['log']),3e-7)
 for r in runs:
  ck('original_mass_shell_relative',r['max_shell_absolute']/K.MN**2,3e-12)
  ck('explicit_decay_inverse_reciprocity',r['max_reciprocity'],3e-12)
 ck('total_mu_anti_KMS',actual[-1]['j']['log']-actual[-1]['chi']['log']+(Enu-(bath.mu_n-bath.mu_p-bath.mu_e))/bath.T,3e-9)
 rows.append(dict(inputs=dict(Enu=Enu,Ee=Ee,cosine=c),independent_runs=runs,kernel_j=actual[-1]['j'],kernel_chi=actual[-1]['chi'],above_rest_endpoint=Enu>((K.MN-K.MP-K.ME)*(K.MN+K.MP+K.ME)/(2*K.MN))))
bad=K.differential(bath,.85,.9,-.7);assert bad['j']['reason']=='empty_support'and bad['chi']['reason']=='empty_support'
endpoint=(K.MN-K.MP-K.ME)*(K.MN+K.MP+K.ME)/(2*K.MN);assert rows[-1]['above_rest_endpoint']and rows[-1]['kernel_j']['value']>0
source=json.loads((ROOT/'relativistic_crossed_thermal/results.json').read_text());ck('reported_selected_checks',sum(not c['passed']for c in source['checks']),0.)
try:K.spectral(bath,.82);rejected=False
except ValueError:rejected=True
assert rejected # Integrated rule restriction, despite physical differential support.
deep=K.Bath(bath.T,bath.mu_n,bath.mu_p,-1000.)
dd=K.differential(deep,.27,1.023,.15,energy_order=24,phi_order=8,tail=80.);rr=independent(deep,.27,1.023,.15,order=20)
assert dd['chi']['value']is None and dd['chi']['reason']=='underflow'and math.isfinite(dd['chi']['log'])
for name in('j','chi'):ck('underflow_finite_log_original_delta_'+name,dd[name]['log']-rr[name],3e-7)
manifest=json.loads((ROOT/'relativistic_crossed_thermal/MANIFEST.json').read_text())
for path,sha in manifest['upstream'].items():ck('frozen_upstream_SHA_'+path,int(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha),0.)
for path,sha in manifest['files'].items():ck('frozen_author_SHA_'+path,int(hashlib.sha256((ROOT/'relativistic_crossed_thermal'/path).read_bytes()).hexdigest()!=sha),0.)
# Reconstruct selected empty-radiation production with one active helicity.
spectra=[row['spectrum']for row in source['rows']if'spectrum'in row];last=spectra[-1];dos=.3**2/(2*math.pi**2*K.eos.HBARC**3);per_n=dos*last['j']['value']/(7e-13)
reported=next(x['cold_limit']['thermal_dGamma_dEnu_per_neutron']for x in source['rows']if'cold_limit'in x and x['cold_limit']['T']==.03)
ck('one_helicity_DOS_per_neutron',per_n/reported-1,3e-10)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),bath=vars(bath),material=material,rows=rows,rest_endpoint=endpoint,underflow=dict(bath=vars(deep),kernel_j=dd['j'],kernel_chi=dd['chi'],independent_logs=rr,scope='Prescribed nonneutral representation test, not matched EOS'),selected_spectrum_per_neutron=per_n,seconds=time.monotonic()-start,scope='Fresh dilute npe differential and above-rest-endpoint probes; no total-energy-integrated lifetime or generic spectral domain certificate')
(P/'results.json').write_text(json.dumps(out,indent=2));print(out['passed'],out['failed'],out['seconds'])
if out['failed']:raise SystemExit(1)
