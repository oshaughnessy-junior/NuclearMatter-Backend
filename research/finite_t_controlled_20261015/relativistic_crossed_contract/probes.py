from pathlib import Path
import sys,json,time,math,hashlib,atexit
from scipy.integrate import quad
from scipy.special import log_expit
sys.dont_write_bytecode=True
from beta import *
HERE=Path(__file__).resolve().parent;start=time.monotonic();checks=[];rows=[];phase='start'
def save():(HERE/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks),phase=phase,checks=checks,rows=rows,runtime_seconds=time.monotonic()-start,source_sha256=hashlib.sha256((HERE/'beta.py').read_bytes()).hexdigest()),indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item())+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for model in ['full','point_VA']:
 vals=[]
 for order in [16,32,48]:
  phase=f'{model}{order}';a=pair_integral(order,model=model);b=electron_integral(order,model=model);vals.append([a,b]);rows.append(dict(model=model,order=order,pair=a,electron=b));save()
  for field in ['rate','energy_neutrino','energy_electron']:gate(f'{model}{order} independent phase {field}',a[field]/b[field]-1,3e-9)
 for j in [0,1]:
  for field in ['rate','energy_neutrino','energy_electron']:gate(f'{model} refinement method{j} {field}',vals[1][j][field]/vals[2][j][field]-1,3e-10)
 gate(model+' full rest-energy ledger',(MN*vals[-1][0]['rate']-vals[-1][0]['energy_neutrino']-vals[-1][0]['energy_electron']-vals[-1][0]['energy_proton'])/(MN*vals[-1][0]['rate']),3e-13)
# Same selected events: reciprocal explicit spinor amplitude and thermal forward/inverse weights.
e=np.array([.68,.91,1.19]);c=np.array([-.43,.27,.61]);pn,pp,l,k,*_=electron_events(e,c,MN,MP,ME)
a=amplitude(pn,pp,l,k);inverse=amplitude(pn,pp,l,k,inverse=True);gate('explicit reciprocal crossed amplitude',np.max(abs(inverse/a-1)),3e-13)
q=k+l;gate('inverse ordered positive timelike q',np.max(abs((pn-pp)-q))/MN,3e-15)
gate('timelike pair mass lower bound',min(0.,float(np.min(q[:,0]**2-np.sum(q[:,1:]**2,axis=1)-ME*ME))),3e-15)
for T,mup,mun,mue in [(2.,930.,933.,35.),(.3,939.,941.,.7)]:
 # j anti: neutron decay fn(1-fp)(1-fe); chi anti: fp fe(1-fn).
 lj=log_expit((mun-pn[:,0])/T)+log_expit((pp[:,0]-mup)/T)+log_expit((l[:,0]-mue)/T)
 lc=log_expit((mup-pp[:,0])/T)+log_expit((mue-l[:,0])/T)+log_expit((pn[:,0]-mun)/T)
 expected=-(k[:,0]-(mun-mup-mue))/T;gate(f'{T} anti KMS from independent products',np.max(abs(lj-lc-expected)),5e-13)
 rows.append(dict(thermal_product_probe=dict(T=T,mup=mup,mun=mun,mue=mue,anti_zeta=mun-mup-mue,logj_product=lj,logchi_product=lc)))
# Exact static vacuum allowed integral is a limit, not a measured lifetime fit.
Q=MN-MP;pref=GF**2*VUD**2*(1+3*1.2723**2)/(2*math.pi**3*HBAR)
static=quad(lambda E:pref*E*math.sqrt((E-ME)*(E+ME))*(Q-E)**2,ME,Q,epsabs=1e-16,epsrel=1e-12)[0]
for M in [1e4,1e5,1e6]:
 mn=M+Q/2;mp=M-Q/2;r=electron_integral(32,mn,mp,ME,'point_VA');error=r['rate']/static-1;gate(f'heavy common bare{M} matched pointVA static limit',error,8e-4 if M==1e4 else 8e-5 if M==1e5 else 8e-6);rows.append(dict(heavy_mass=M,rate=r['rate'],static_rate=static,relative_error=error))
# Smooth spectrum data, physical endpoint, and selected static-limit shape.
Emax=((MN-MP)*(MN+MP)+ME*ME)/(2*MN)
for E in [.56,.75,1.,1.24]:
 f=spectrum(E,16);g=spectrum(E,32);gate('spectrum angular refinement '+str(E),f/g-1,3e-12);rows.append(dict(electron_total_energy=E,full_current_spectrum=g,pointVA_spectrum=spectrum(E,32,model='point_VA')))
rows.append(dict(static_rate=static,static_lifetime=1/static,physical_electron_endpoint=Emax,physical_antineutrino_endpoint=((MN-MP-ME)*(MN+MP+ME))/(2*MN)))
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
