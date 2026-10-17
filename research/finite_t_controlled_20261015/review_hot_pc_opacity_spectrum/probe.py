"""Independent spacelike u-shell / log-positron-momentum PC spectral review."""
from pathlib import Path
import importlib.util,sys,json,math,hashlib,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit,logsumexp
P=Path(__file__).parent;ROOT=P.parent;A=ROOT/'hot_pc_opacity_spectrum';sys.dont_write_bytecode=True
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
S=load('review_hotPC_frozen_spectrum',A/'spectrum.py');O=load('review_hotPC_original_delta_spinors',ROOT/'relativistic_delta_oracle/oracle.py');b,meta=S.matched();C=S.C

def log_delta(E,El,c,n=20,phi=6,tail=70.):
 # In a spacelike-transfer Lorentz frame omega'=0. Original pz delta,
 # cylindrical r dr=mp du, with both shells solved without K.support.
 pl=math.sqrt((El-C.ME)*(El+C.ME));w=E-El;q=math.sqrt(E*E+pl*pl-2*E*pl*c);aa=q*q-w*w
 if aa<=0:return dict(j=None,chi=None)
 rt=math.sqrt(aa);z0=-(C.MN**2-C.MP**2+aa)/(2*rt);emin=(q*math.hypot(C.MP,z0)+w*z0)/rt;hi=max(emin,b.mu_p,b.mu_n-w)+tail*b.T
 def u_at(ea):return(((ea*rt-w*z0)/q)**2-C.MP**2-z0*z0)/(2*C.MP)
 cuts=sorted({0.,max(0.,u_at(hi))}|{u_at(x)for h in[1,4,10,22,40]if emin<(x:=max(emin,b.mu_p,b.mu_n-w)+h*b.T)<hi});x,g=leggauss(n);us=[];ws=[]
 for a,d in zip(cuts[:-1],cuts[1:]):us.extend(a+(x+1)*(d-a)/2);ws.extend(g*(d-a)/2)
 us=np.repeat(np.array(us),phi);weights=np.repeat(np.array(ws),phi)/phi;ph=np.tile(2*math.pi*np.arange(phi)/phi,len(us)//phi);r=np.sqrt(2*C.MP*us);ep=np.sqrt(C.MP**2+r*r+z0*z0);pz=(q*z0+w*ep)/rt;ea=(q*ep+w*z0)/rt;eb=ea+w;delta=abs((pz+q)/eb-pz/ea);pa=np.column_stack([ea,r*np.cos(ph),r*np.sin(ph),pz]);pb=pa+np.array([w,0,0,q]);kz=(E*E+q*q-pl*pl)/(2*q);kx2=E*E-kz*kz
 if kx2< -1e-10*E*E:raise ArithmeticError('Lepton q-axis roundoff')
 kx=math.sqrt(max(0.,kx2));k=np.tile([E,kx,0,kz],(len(us),1));l=np.tile([El,kx,0,kz-q],(len(us),1));amp=S.P.amplitude(k,pa,l,pb)
 if np.any(amp<=0)or np.any(weights<=0):raise ArithmeticError('Independent positive phase failed')
 base=np.log(weights*C.MP*amp*pl/(32*math.pi**3*E*C.HBAR*ea*eb*delta));lf=log_expit((b.mu_n-eb)/b.T)+log_expit((ea-b.mu_p)/b.T)+log_expit((-b.mu_e-El)/b.T);lb=log_expit((b.mu_p-ea)/b.T)+log_expit((eb-b.mu_n)/b.T)+log_expit((El+b.mu_e)/b.T)
 return dict(j=float(logsumexp(base+lf)),chi=float(logsumexp(base+lb)))

def integrated(E,n=8,angular=12,inner=16,phi=6,tail=60.,upper=90.):
 start=time.monotonic();center=E+C.MP-C.MN;p0=math.sqrt(max(0.,center*center-C.ME*C.ME));width=max(.05*b.T,(E+p0)*math.sqrt(b.T/C.MP));cuts=sorted({C.ME,upper}|{v for scale in[width,b.T]for z in[-12,-6,-3,-1.5,-.75,-.25,0,.25,.75,1.5,3,6,12]if C.ME<(v:=center+z*scale)<upper});edges=[math.log1p(math.sqrt(v*v-C.ME*C.ME)/b.T)for v in cuts];x,w=leggauss(n);z,v=leggauss(angular);logs={'j':[],'chi':[]};calls=0
 for lo,hi in zip(edges[:-1],edges[1:]):
  for xx,ww in zip(x,w):
   y=lo+(xx+1)*(hi-lo)/2;pl=b.T*math.expm1(y);El=math.hypot(pl,C.ME);jac=ww*(hi-lo)/2*b.T*math.exp(y)*pl/El;cmax=min(1.,(El-C.ME*C.ME/(2*E))/pl)
   if cmax<=-1:continue
   # Exact spacelike lepton boundary and forward-resolving angle coordinate.
   for zz,vv in zip((z+1)/2,v/2):
    c=cmax-(cmax+1)*zz*zz;cw=2*(cmax+1)*zz*vv;r=log_delta(E,El,c,inner,phi,tail);calls+=1
    if time.monotonic()-start>110:raise TimeoutError('Independent spectral integral110s cap')
    for key in logs:
     if r[key]is not None:logs[key].append(r[key]+math.log(jac*cw))
 return dict(E=E,j=float(logsumexp(logs['j'])),chi=float(logsumexp(logs['chi'])),controls=[n,angular,inner,phi,tail,upper],calls=calls,seconds=time.monotonic()-start)

def main():
 before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.iterdir()if p.is_file()};checks=[];points=[]
 def ck(name,er,gate=3e-10):checks.append(dict(name=name,error=float(er),gate=gate,passed=bool(math.isfinite(er)and abs(er)<=gate)))
 ck('commissioned source SHA',0 if before['hot_pc_opacity_spectrum/spectrum.py']=='5fbc8fc21fc6b3b26bf43331ebe04b31bf66145b1f7bf51c977dfa47e5f78b0b'else 1,0);ck('commissioned manifest SHA',0 if before['hot_pc_opacity_spectrum/MANIFEST.json']=='1c60b6e64e31aecc32684a50c780182c176e136d05fea854d8791caf62eaf323'else 1,0)
 for en,el,c in[(1.73,.574,-.31),(6.13,4.83,.47),(12.37,10.99,-.61)]:
  own=log_delta(en,el,c,24,8,80.);d=S.differential(b,en,el,c,energy_order=24,phi_order=8,tail=80.);p=O.Point(b.T,C.MP,C.MN,b.mu_p,b.mu_n,en,el,c,C.ME,-b.mu_e,True,C.MW);oracle=O.integrate(p,order=36,nphi=10,tail=80.)
  for k in['j','chi']:ck('new original-delta spinor '+str((en,k)),math.expm1(d[k]['log']-math.log(oracle[k])));ck('independent logdelta '+str((en,k)),math.expm1(own[k]-d[k]['log']))
  points.append(dict(inputs=[en,el,c],own=own,source=d,spinor={k:oracle[k]for k in['j','chi','max_shell_residual']}))
 for en,el,c in[(1.31,4.73,.9997),(3.19,11.83,.9998)]:
  d=S.differential(b,en,el,c);ck('pasttimelike exact empty '+str(en),0 if d['j']['log']is None and d['chi']['log']is None and d['j']['value']==d['chi']['value']==0 else 1,0)
 out=dict(checks=checks,points=points,rows=[],before=before)
 if any(not z['passed']for z in checks):raise RuntimeError('Absolute forward anchors failed')
 for controls in[(8,12,16,6,60.,90.),(12,18,24,8,85.,110.)]:
  r=integrated(3.47,*controls);out['rows'].append(r);(P/'partial.json').write_text(json.dumps(out,indent=2));print('alternate phase',r,flush=True)
 actual=S.integrate(3.47,lepton_order=12,angle_order=20,energy_order=16,phi_order=8,tail=70.,outer_tail=40.);out['fresh_source_reference']=actual
 for k in['j','chi']:
  ck('alternate integration refinement '+k,math.expm1(out['rows'][0][k]-out['rows'][1][k]),3e-6);ck('new E3.47 alternate vs source '+k,math.expm1(out['rows'][1][k]-actual[k]['log']),3e-6)
 after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.iterdir()if p.is_file()};ck('author bytes unchanged',0 if after==before else 1,0);out.update(passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],after=after);(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed']),indent=2))
if __name__=='__main__':main()
