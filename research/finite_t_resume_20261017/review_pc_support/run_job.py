from pathlib import Path
import sys,json,hashlib,ast,math,time,types,dataclasses
import numpy as np
from scipy.special import roots_genlaguerre,roots_jacobi,logsumexp
from numpy.polynomial.legendre import leggauss
P=Path(__file__).resolve().parent;R=P.parent;OLD=R.parent/'nmb-finite-t-20261015';L=json.loads((P/'LOCK.json').read_text());job=L['jobs'][int(sys.argv[1])];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();idx=int(sys.argv[1]);out=P/f'job{idx}.json';assert not out.exists()
def verify():
 for n,h in L['author_files'].items():assert sha(R/'pc_support'/n)==h,n
verify();start=time.monotonic()
if job['kind']=='candidate':
 sys.path.insert(0,str(R/'pc_support'));import conditional_pc as C
 z=C.evaluate(job['E'],C.Rule(radial=job['radial'],incoming=job['incoming'],outgoing=job['outgoing'],phi=job['phi'],radial_family='laguerre',budget=85.));verify();out.write_text(json.dumps(dict(kind='fresh candidate heldout',job=job,result=z,seconds=time.monotonic()-start),indent=2)+'\n');print(idx,'candidate',z['log_j_s-1'],z['log_true_chi_s-1'],time.monotonic()-start);sys.exit()
# Independent previously frozen explicit-spinor implementation; select definitions
# only. Never import new candidate contraction or original top-level driver.
src=OLD/'relativistic_capture_oracle/events.py';prep=json.loads((R/'independent_review_preparation/LOCK.json').read_text());assert sha(src)==prep['sourcepins']['relativistic_capture_oracle/events.py'];tree=ast.parse(src.read_text());allowed={'GF','VUD','HBAR','HC','S','Z','G','G5','I','MET'};defs=[]
for n in tree.body:
 if isinstance(n,ast.Assign)and all(isinstance(t,ast.Name)and t.id in allowed for t in n.targets):defs.append(n)
 if isinstance(n,ast.FunctionDef)and n.name in['spinors','adj','vertices','amplitude','boost']:defs.append(n)
NS={'np':np};exec(compile(ast.Module(body=defs,type_ignores=[]),str(src)+'<selected-old-spinors>','exec'),NS)
MN=939.56542052;MP=938.27208816;ME=.51099895069;T=2.;mun=919.8287578100662;mup=916.8449303084382;mue=.17446333200826397;E=job['E'];HBAR=6.582119569e-22;b=types.SimpleNamespace(ma=MP,mb=MN,ml=ME,anti=True,Mweak=(MN+MP)/2)
ds0=(MN+ME-MP)*(MN+ME+MP);sth=(MN+ME)**2;K=ds0/(2*E);E0=(K*K+MP*MP)/(2*K);p0=(K*K-MP*MP)/(2*K)
x,wx=roots_genlaguerre(job['radial'],1.5);a,wa=roots_jacobi(job['incoming'],0.,.5);xi=(a+1)/2;iw=wa/(2**1.5*np.sqrt(xi));co,wo=leggauss(job['outgoing']);phi=2*math.pi*np.arange(job['phi'])/job['phi'];XI,CO,PH=np.meshgrid(xi,co,phi,indexing='ij');WW=(iw[:,None,None]*wo[None,:,None]*np.ones((1,1,job['phi']))*2*math.pi/job['phi']);XI,CO,PH,WW=[v.ravel()for v in[XI,CO,PH,WW]];logs=[[],[]];mass=delta=four=0.;events=0
for x0,w0 in zip(x,wx):
 if time.monotonic()-start>85:raise TimeoutError('Independent oracle budget85s, no complete output')
 de=T*x0;Ep=E0+de;p=math.sqrt((Ep-MP)*(Ep+MP));A=de*(1+(Ep+E0)/(p+p0));cap=A/p;assert 0<cap<2
 cp=-1+cap*(1-XI);sn=np.sqrt(cap*(1-XI)*(2-cap*(1-XI)));proton=np.column_stack([np.full(len(XI),Ep),p*sn,np.zeros(len(XI)),p*cp]);k=np.tile([E,0.,0.,E],(len(XI),1));s=sth+2*E*A*XI;rt=np.sqrt(s);pf=np.sqrt((2*E*A*XI)*(2*E*A*XI+4*MN*ME))/(2*rt);pi=(ds0+2*E*A*XI)/(2*rt);total=proton+k;boostv=total[:,1:]/total[:,0,None]
 kstar=NS['boost'](k,-boostv);axis=kstar[:,1:]/np.linalg.norm(kstar[:,1:],axis=1)[:,None];ey=np.tile([0.,1.,0.],(len(XI),1));ex=np.cross(ey,axis);direc=CO[:,None]*axis+np.sqrt(1-CO*CO)[:,None]*(np.cos(PH)[:,None]*ex+np.sin(PH)[:,None]*ey)
 Elstar=(s+ME*ME-MN*MN)/(2*rt);Enstar=(s+MN*MN-ME*ME)/(2*rt);lepton=NS['boost'](np.column_stack([Elstar,pf[:,None]*direc]),boostv);neutron=NS['boost'](np.column_stack([Enstar,-pf[:,None]*direc]),boostv)
 amp=NS['amplitude'](b,k,proton,lepton,neutron);assert np.all(amp>0)and np.all(np.isfinite(amp))
 # Keep original invariant moving-target flux/cross-section normalization,
 # with avg2 inverse current and target spin2; do NOT borrow candidate avg4.
 flux=(ds0+2*E*A*XI)/(2*E*Ep);dsigma=amp*pf/pi/(64*math.pi**2*s)
 base=math.log(2/(4*math.pi**2*HBAR))+math.log(p*Ep*T*w0)+x0-1.5*math.log(x0)+math.log(cap)+np.log(WW)+np.log(flux)+np.log(dsigma)
 lf=lambda energy,mu:-np.logaddexp(0.,(energy-mu)/T)
 chi=lf(Ep,mup)+lf(-neutron[:,0],-mun)+lf(-lepton[:,0],mue)
 jj=lf(neutron[:,0],mun)+lf(lepton[:,0],-mue)+lf(-Ep,-mup)
 logs[0].append(float(logsumexp(base+jj)));logs[1].append(float(logsumexp(base+chi)))
 invariant=lambda v:v[:,0]**2-np.sum(v[:,1:]**2,axis=1)
 mass=max(mass,float(np.max(abs(invariant(neutron)/MN**2-1))),float(np.max(abs(invariant(lepton)/ME**2-1))))
 four=max(four,float(np.max(abs(total-neutron-lepton)))/Ep)
 ev=proton[:,1:]+k[:,1:]-lepton[:,1:];es=np.sqrt(MN**2+np.sum(ev*ev,axis=1));delta=max(delta,float(np.max(abs(Ep+E-lepton[:,0]-es)))/Ep)
 assert mass<2e-11 and four<2e-12 and delta<2e-12
 events+=len(XI)
lj,lc=[float(logsumexp(v))for v in logs];z=dict(log_j=lj,log_chi=lc,ordinary_j=None if lj<math.log(np.nextafter(0.,1.))else math.exp(lj),ordinary_chi=None if lc<math.log(np.nextafter(0.,1.))else math.exp(lc),KMS_posterior=lj-lc+(E-(mun-mup-mue))/T,threshold_Ep=E0,events=events,max_shell=mass,max_four=four,max_unsquared_delta=delta,radial_domain='semi-infinite',coordinates='threshold targetenergy/genLaguerre alpha1.5 + original cap xi/Jacobi beta.5; nu lab z and proton conditional polarcap',current='old independent literal-spinor inverse avg2; no candidate/current import',normalization='unreduced g2target measure × Moellerflux × physical dσ/dΩ*/hbar',seconds=time.monotonic()-start)
verify();out.write_text(json.dumps(dict(job=job,result=z,source_sha256=sha(src),selected_AST_functions=['spinors','adj','vertices','amplitude','boost'],complete=True),indent=2,allow_nan=False)+'\n');print(idx,'oracle',lj,lc,z['seconds'])
