from pathlib import Path
import sys,ast,math,json,time,hashlib
import numpy as np
from scipy.special import roots_genlaguerre,roots_jacobi,logsumexp
from numpy.polynomial.legendre import leggauss
P=Path(__file__).resolve().parent;R=P.parent;O=R.parent/'nmb-finite-t-20261015';L=json.loads((P/'LOCK.json').read_text());idx=int(sys.argv[1]);J=L['jobs'][idx];out=P/f'job{idx}.json';assert not out.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 for f,h in L['author_files'].items():assert sha(R/'beta_support'/f)==h,f
verify();start=time.monotonic()
if J['kind']=='candidate':
 sys.path.insert(0,str(R/'beta_support'));import conditional_beta as C
 z=C.evaluate(J['E'],C.Rule(J['parent'],J['rest_nu'],J['electron'],J['phi'],'laguerre',budget=85.));verify();out.write_text(json.dumps({'kind':'fresh candidate','job':J,'result':z},indent=2)+'\n');print(idx,'candidate',z['log_j_s-1'],z['log_true_chi_s-1']);sys.exit()
MN=939.56542052;MP=938.27208816;ME=.51099895069;T=2.;mun=919.8287578100662;mup=916.8449303084382;mue=.17446333200826397;HC=197.3269804;HBAR=6.582119569e-22;E=J['E'];N={'np':np,'math':math,'MN':MN,'MP':MP,'ME':ME}
src=O/'relativistic_capture_oracle/events.py';beta=O/'relativistic_crossed_contract/beta.py';pins=json.loads((R/'beta_review_preparation/LOCK.json').read_text())['source_sha256'];assert sha(src)==pins[str(src)] and sha(beta)==pins[str(beta)]
allowed={'GF','VUD','HBAR','HC','S','Z','G','G5','I','MET'};defs=[]
for n in ast.parse(src.read_text()).body:
 if isinstance(n,ast.Assign)and all(isinstance(t,ast.Name)and t.id in allowed for t in n.targets):defs.append(n)
 if isinstance(n,ast.FunctionDef)and n.name in ('spinors','adj','vertices','boost'):defs.append(n)
exec(compile(ast.Module(body=defs,type_ignores=[]),str(src)+'<spinors-only>','exec'),N)
f=[n for n in ast.parse(beta.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='amplitude'];exec(compile(ast.Module(body=f,type_ignores=[]),str(beta)+'<amplitude-only>','exec'),N)
E0=(MN-MP-ME)*(MN+MP+ME)/(2*MN);D=MN*E0/E;A0=(MN*MN+D*D)/(2*D);p0=(MN*MN-D*D)/(2*D)
xs,ws=roots_genlaguerre(J['parent'],1.5);a,w=roots_jacobi(J['rest_nu'],0.,.5);zeta=(a+1)/2;wz=w/(2**1.5*np.sqrt(zeta));xe,we=leggauss(J['electron']);ph=2*math.pi*np.arange(J['phi'])/J['phi'];logs=[[],[]];ms=fs=ds=0.;events=0;recip=0.
for x,wx in zip(xs,ws):
 if time.monotonic()-start>85:raise TimeoutError('Independent restDalitz oracle incomplete')
 de=T*x;A=A0+de;p=math.sqrt((A-MN)*(A+MN));g=A/MN;v=p/A;B=E0*de*(1+(A+A0)/(p+p0))/(A+p)
 ev=E0-B*zeta;wev=B*wz;cnu=(E/(g*ev)-1)/v;assert np.all(abs(cnu)<1)
 S=(MP+ME)**2+2*MN*B*zeta;rt=np.sqrt(S);ps=np.sqrt((2*MN*B*zeta)*(2*MN*B*zeta+4*MP*ME))/(2*rt);estar=np.hypot(ps,ME);gp=(MN-ev)/rt;vp=ev/(MN-ev)
 ee=gp[:,None]*(estar[:,None]+vp[:,None]*ps[:,None]*xe[None,:]);wee=gp[:,None]*vp[:,None]*ps[:,None]*we[None,:]
 evv=np.broadcast_to(ev[:,None],ee.shape);cc=np.broadcast_to(cnu[:,None],ee.shape)
 # Electron rest energy, not pair-CM angle: recover longitudinal momentum
 # from the on-shell unsquared n->p+e+nu energy constraint.
 pz=(MN*MN+ME*ME-MP*MP-2*MN*(ee+evv)+2*ee*evv)/(2*evv)
 px2=(ee-ME)*(ee+ME)-pz*pz;assert np.all(px2>0)
 lepton=np.column_stack([ee.ravel(),np.sqrt(px2).ravel(),np.zeros(ee.size),pz.ravel()]);nu=np.column_stack([evv.ravel(),np.zeros(ee.size),np.zeros(ee.size),evv.ravel()]);nn=np.tile([MN,0.,0.,0.],(ee.size,1));proton=nn-lepton-nu
 amp=N['amplitude'](nn,proton,lepton,nu);assert np.all(amp>0)
 if events==0:
  inv=N['amplitude'](nn,proton,lepton,nu,inverse=True);recip=float(np.max(abs(inv/amp-1)));assert recip<1e-10
 # Independent original rest-Dalitz measure, proper-time, conditional cone,
 # spin2 parent density, and divide ONE active-helicity DOS.
 parent=wx*T*p*A/(math.pi**2*HC**3)
 dos=E*E/(2*math.pi**2*HC**3)
 base=math.log(parent/g/dos)+x-1.5*math.log(x)+np.log(amp/(64*math.pi**3*MN*HBAR))
 base+=np.log((wev[:,None]*wee/(2*g*v*ev[:,None])).ravel())-math.log(J['phi'])
 def rotate_boost(a):
  c=cc.ravel()[:,None];sn=np.sqrt((1-c)*(1+c));xx=a[:,1,None]*np.cos(ph)-a[:,2,None]*np.sin(ph);yy=a[:,1,None]*np.sin(ph)+a[:,2,None]*np.cos(ph);zz=-sn*xx+c*a[:,3,None];xx=c*xx+sn*a[:,3,None];en=g*(a[:,0,None]+v*zz);zz=g*(zz+v*a[:,0,None]);return np.stack([en,xx,yy,zz],axis=-1)
 labl=rotate_boost(lepton);labp=rotate_boost(proton);labk=rotate_boost(nu);ln=np.zeros_like(labl);ln[:,:,0]=A;ln[:,:,3]=p
 inv4=lambda a:a[:,:,0]**2-np.sum(a[:,:,1:]**2,axis=-1)
 ms=max(ms,float(np.max(abs(inv4(labl)/ME**2-1))),float(np.max(abs(inv4(labp)/MP**2-1))))
 fs=max(fs,float(np.max(abs(ln-labl-labp-labk)))/A,float(np.max(abs(labk[:,:,0]-E)))/E)
 prvec=ln[:,:,1:]-labl[:,:,1:]-labk[:,:,1:];shell=np.sqrt(MP**2+np.sum(prvec*prvec,axis=-1));ds=max(ds,float(np.max(abs(A-labl[:,:,0]-E-shell)))/A)
 assert ms<2e-11 and fs<2e-12 and ds<2e-12 and np.all(labl[:,:,0]>=ME) and np.all(labp[:,:,0]>=MP)
 lf=lambda energy,mu:-np.logaddexp(0.,(energy-mu)/T)
 jj=lf(A,mun)+lf(-labp[:,:,0],-mup)+lf(-labl[:,:,0],-mue);ch=lf(-A,-mun)+lf(labp[:,:,0],mup)+lf(labl[:,:,0],mue)
 logs[0].append(float(logsumexp(base[:,None]+jj)));logs[1].append(float(logsumexp(base[:,None]+ch)));events+=labl.shape[0]*labl.shape[1]
lj,lc=[float(logsumexp(v))for v in logs];verify();r=dict(log_j=lj,log_chi=lc,j=math.exp(lj),chi=math.exp(lc),threshold_En=A0,events=events,max_shell=ms,max_four=fs,max_unsquared_delta=ds,reciprocal_literal=recip,KMS_posterior=lj-lc+(E-(mun-mup-mue))/T,seconds=time.monotonic()-start,radial_domain='semi-infinite threshold-shifted',current='old explicit crossed-contract spinors, not candidate current import',coordinates='rest neutrino energy Jacobi; rest electron Dalitz energy; relative azimuth; original conditional lab-energy cone')
out.write_text(json.dumps({'job':J,'result':r,'complete':True,'old_spinor_sha256':sha(src),'old_beta_sha256':sha(beta)},indent=2)+'\n');print(idx,'oracle',lj,lc,r['seconds'])
