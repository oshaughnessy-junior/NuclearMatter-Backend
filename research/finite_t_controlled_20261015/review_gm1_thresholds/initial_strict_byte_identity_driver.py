"""Independent beta solve in mu-space, nested triangle roots and geometry."""
from pathlib import Path
import sys,math,json,hashlib,time
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'gm1_beta_thresholds'));import cold_beta as B
GM=B.gm;H=GM.HBARC;ME=B.ME;MM=B.MMU
checks=[]
def ck(name,e,t):checks.append(dict(name=name,error=abs(float(e)),limit=t,passed=bool(abs(e)<=t)))

def density(mu,m):
 if mu<=m:return 0.
 p=math.sqrt(mu*mu-m*m)
 return quad(lambda x:x*x,0,p,epsabs=1e-15,epsrel=2e-13)[0]/(math.pi**2*H**3)

def independent(n,me=ME):
 # Different unknown/root equation: charge gives Yp(mu), then baryon affinity.
 def yp(mu):return(density(mu,me)+density(mu,MM))/n
 upper=brentq(lambda mu:yp(mu)-.49,me+1e-3,1500.)
 def f(mu):
  nuc=GM.cold_state(n,yp(mu))
  return mu-nuc.mu_n_mev+nuc.mu_p_mev
 mu=brentq(f,5.,upper,xtol=8e-13,rtol=2e-14)
 y=yp(mu);nuc=GM.cold_state(n,y);ne=density(mu,me);nm=density(mu,MM)
 k=lambda d:(3*math.pi**2*d)**(1/3)*H
 kn,kp,ke,km=map(k,(n*(1-y),n*y,ne,nm))
 return dict(nB=n,Yp=y,Ye=ne/n,Ymu=nm/n,mu=mu,nuclear=nuc,delta_e=kn-kp-ke,delta_mu=kn-kp-km,momenta=[kn,kp,ke,km],root_derivative=(f(mu+1e-3)-f(mu-1e-3))/2e-3,root_bracket=[5.,upper],affinity_residual=f(mu))

t0=time.monotonic();samples=[]
for n in(.105,.21,.31,.55):
 r=independent(n);b=B.state(n);samples.append(r)
 for name,a,c in [('Yp',r['Yp'],b.Yp),('Ye',r['Ye'],b.Ye),('Ymu',r['Ymu'],b.Ymu)]:ck('independent_'+name,a-c,2e-11)
 ck('independent_mu',r['mu']-b.mu_lepton,2e-8)
 ck('mu_space_affinity',r['affinity_residual'],2e-9)
 assert r['root_derivative']>1
 # Independent chemical finite differences at fixed species densities.
 nn=n*(1-r['Yp']);np_=n*r['Yp'];cols=[]
 for species in(0,1):
  step=[nn,np_][species]*1e-5;v=[]
  for sign in(-1,1):
   den=np.array([nn,np_]);den[species]+=sign*step;z=GM.cold_state(sum(den),den[1]/sum(den));v.append(np.array([z.mu_n_mev,z.mu_p_mev]))
  cols.append((v[1]-v[0])/(2*step))
 h=np.array(cols).T;author=np.array(b.nuclear_hessian)
 ck('independent_cold_Hessian',np.max(abs(h-author))/np.max(abs(author)),3e-6)
 ck('Hessian_Maxwell',abs(h[0,1]-h[1,0])/np.max(abs(h)),3e-6)
 assert np.linalg.eigvalsh((h+h.T)/2)[0]>0
 # Differentiate the neutral beta branch: dP/dn=n*dmu_n/dn, dE/dn=mu_n.
 step=n*2e-5;branch=[B.state(n+sign*step)for sign in(-1,1)]
 dE=(branch[1].energy_density-branch[0].energy_density)/(2*step);dP=(branch[1].pressure-branch[0].pressure)/(2*step);dm=(branch[1].nuclear.mu_n_mev-branch[0].nuclear.mu_n_mev)/(2*step)
 ck('beta_branch_dE_dn',dE/b.nuclear.mu_n_mev-1,3e-7)
 ck('beta_branch_dP_dn',dP/(n*dm)-1,3e-7)

onsets={}
for kind,bracket in[('appearance',(.11,.15)),('e',(.24,.33)),('mu',(.30,.42))]:
 def f(n):
  r=independent(n)
  return r['mu']-MM if kind=='appearance'else r['delta_'+kind]
 root=brentq(f,*bracket,xtol=5e-13,rtol=2e-14);r=independent(root);onsets[kind]=r
 if kind!='appearance':
  author=B.threshold(kind,*( (.26,.30)if kind=='e'else(.32,.38)))
  ck('independent_onset_density_'+kind,root-author.nB,2e-10)
  ck('independent_onset_triangle_'+kind,r['delta_'+kind],2e-7)
  sides=r['momenta'];kn,kp,kl=sides[0],sides[1],sides[2 if kind=='e' else 3]
  x=(r['Ye']if kind=='e'else r['Ymu'])/r['Yp'];ypcrit=1/(1+(1+x**(1/3))**3)
  ck('triangle_fraction_identity_'+kind,r['Yp']-ypcrit,2e-10)
  ck('collinear_surface_angle_'+kind,(kn*kn-kp*kp-kl*kl)/(2*kp*kl)-1,2e-8)
  below=independent(root-1e-4);above=independent(root+1e-4)
  assert below['delta_'+kind]>0>above['delta_'+kind]
  ka,kb,kc=above['momenta'][0],above['momenta'][1],above['momenta'][2 if kind=='e' else 3]
  c=(ka*ka-kb*kb-kc*kc)/(2*kb*kc);assert -1<c<1
  vector=np.array([kc*math.sqrt(1-c*c),0.,kb+kc*c]);ck('allowed_surface_vector_'+kind,np.linalg.norm(vector)-ka,2e-10)
  r['bracket_slope']=(above['delta_'+kind]-below['delta_'+kind])/2e-4
 else:
  ck('appearance_contact_mu',r['mu']-MM,2e-8)
  assert independent(root-1e-5)['Ymu']==0 and independent(root+1e-5)['Ymu']>0

# Exact source-byte parity provenance, without rerunning or editing author JAX.
pinned=json.loads((ROOT/'gm1_beta_thresholds/pinned_results.json').read_text())
for path,sha in pinned['sources'].items():
 ck('pinned_source_SHA_'+path,int(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha),0.)
assert pinned['mass_convention']['pinned_electron']==.510998950
assert pinned['mass_convention']['own_default_electron']==ME
assert pinned['mass_convention']['common_bare_nucleon']==939.
ck('reported_pinned_primal_checks',sum(not c['passed']for c in pinned['checks']),0.)
meta=json.loads((P/'github_refetch.json').read_text())
for item,path in zip(meta['files'],['gm1_beta_thresholds/source_snapshot/leptons.py','rmf_reference/source_snapshot/rmft.py','rmf_reference/source_snapshot/constants.py']):
 data=(ROOT/path).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
 ck('independent_GitHub_blob_SHA_'+path,int(blob!=item['sha']),0.)
def lepton_energy(mu,m):
 if mu<=m:return 0.
 p=math.sqrt(mu*mu-m*m)
 return(p*mu*(2*p*p+m*m)-m**4*math.asinh(p/m))/(8*math.pi**2*H**3)
for row in pinned['rows']:
 r=independent(row['nB'],me=.510998950);v=row['pinned_source'];n=r['nuclear']
 energy=n.energy_density+lepton_energy(r['mu'],.510998950)+lepton_energy(r['mu'],MM)
 pressure=GM.cold_state(row['nB'],r['Yp']).pressure+r['mu']*row['nB']*r['Yp']-lepton_energy(r['mu'],.510998950)-lepton_energy(r['mu'],MM)
 for key,x in [('Yp',r['Yp']),('mu',r['mu']),('Ye',r['Ye']),('Ymu',r['Ymu']),('Phi',n.phi_mev),('energy',energy),('pressure',pressure),('mu_n',n.mu_n_mev)]:ck('independent_matched_legacy_'+key,(x-v[key])/max(abs(x),1e-10),3e-9)
warm=json.loads((ROOT/'gm1_beta_thresholds/results.json').read_text())
warm_checks=[]
def pair_density(T,mu,m):
 end=max(m,abs(mu))+50*T;pmax=math.sqrt((end-m)*(end+m))
 return quad(lambda p:p*p*(expit((mu-math.hypot(p,m))/T)-expit((-mu-math.hypot(p,m))/T)),0,pmax,epsabs=1e-6,epsrel=2e-11,limit=200)[0]/(math.pi**2*H**3)
for row in warm['warm']:
 s=row['matter'];n=s['nuclear'];T=n['temperature'];nu={}
 for kind,U,target in [('n',n['omega_mev']-.5*n['rho_mev'],n['nB']*(1-n['Yp'])),('p',n['omega_mev']+.5*n['rho_mev'],n['nB']*n['Yp'])]:
  nu[kind]=n['mu_'+kind+'_mev']-U;measured=pair_density(T,nu[kind],n['dirac_mass_mev']);ck('warm_independent_baryon_density',measured/target-1,2e-9)
 affinity={}
 for kind,key,m in [('e','electron',ME),('mu','muon',MM)]:
  measured=pair_density(T,s[key]['chemical_potential'],m);target=n['nB']*s['Y'+('e'if kind=='e'else'mu')]
  ck('warm_independent_lepton_density',measured/target-1,2e-9)
  affinity[kind]=s[key]['chemical_potential']+n['mu_p_mev']-n['mu_n_mev']
  assert abs(affinity[kind])>1e-3
  ck('fixed_cold_fraction',s['Y'+('e'if kind=='e'else'mu')]-onsets[row['cold_onset']]['Y'+('e'if kind=='e'else'mu')],2e-11)
 warm_checks.append(dict(cold_onset=row['cold_onset'],temperature=T,affinities=affinity,scope='Fixed cold fractions, independently positive-T density checked; NOT warm beta equilibrium or weak rates'))
def encode(x):
 if hasattr(x,'asdict'):return x.asdict()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),samples=samples,onsets=onsets,warm_checks=warm_checks,seconds=time.monotonic()-t0,pinned_scope='Independent exact-ref GitHub blob verification and mu-space matched legacy comparison to selected-primal report; not an independent rerun of JAX or external NMB package')
(P/'results.json').write_text(json.dumps(out,indent=2,default=encode));print(out['passed'],out['failed'],out['seconds'])
if out['failed']:raise SystemExit(1)
