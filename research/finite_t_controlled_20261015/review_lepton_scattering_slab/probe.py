"""Independent finite-input slab review; only this review leaf is written."""
from pathlib import Path
import json,sys,types,time,hashlib,math
import numpy as np
from scipy.integrate import quad,solve_ivp
from scipy.special import expit
HERE=Path(__file__).resolve().parent;AUTHOR=HERE.parent/'lepton_scattering_slab'
sys.dont_write_bytecode=True
B=types.ModuleType('frozen_reviewed_slab');B.__file__=str(AUTHOR/'slab.py')
exec(compile((HERE/'reviewed_initial_slab.py').read_bytes(),str(HERE/'reviewed_initial_slab.py'),'exec'),B.__dict__)
checks=[];start=time.monotonic()
def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(np.isfinite(error)and abs(error)<=tol)))
def save(name,data):(HERE/name).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item())+'\n')

def own_D(E,F,c,pos,tail=90.):
    # Analytic target azimuth, adaptive target ENERGY; does not call author
    # scatter.py amplitude/quad or slab reference.py transverse-shell recipe.
    M=.51099895069;GF=1.1663785e-11;HB=6.582119569e-22
    om=E-F;s=2*E*F*(1-c);Q=math.sqrt(om*om+s)
    lo=(-om+Q*math.sqrt(1+4*M*M/s))/2
    hi=max(lo,B.MU,-B.MU,abs(om))+tail*2.
    mut=-B.MU if pos else B.MU;cv=.96258;ca=.5
    L,R=(cv+ca)**2,(cv-ca)**2
    if pos:L,R=R,L
    kz=(E*E+Q*Q-F*F)/(2*Q);kx2=max(0.,E*E-kz*kz);h=s/2
    def target(Et):
      pz=(2*Et*om-s)/(2*Q);pt2=max(0.,Et*Et-M*M-pz*pz);u=E*Et-kz*pz
      A=L*(u*u+kx2*pt2/2)+R*((u-h)**2+kx2*pt2/2)-M*M*(cv*cv-ca*ca)*h
      return A*expit((mut-Et)/2)*expit((Et+om-mut)/2)
    integral=quad(target,lo,hi,epsabs=1e-9,epsrel=2e-11,limit=120)[0]
    return F*16*GF*GF/(64*math.pi**3*E*Q*HB)*integral

def own_edge(E,F,u,v,tail=90):
    def az(phi):
      c=u*v+math.sqrt((1-u*u)*(1-v*v))*math.cos(phi)
      return sum(own_D(E,F,c,p,tail)for p in (False,True))/math.pi
    # reflection symmetry: average over 0..pi, adaptive not midpoint samples.
    return quad(az,0,math.pi,epsabs=2e-11,epsrel=3e-9,limit=120)[0]

def independent_rhs(g,ff,left,right,cross):
    E=np.array(g['E']);W=np.array(g['W']);m=np.array(g['mu']);A=np.array(g['A']);nc,N=ff.shape
    df=np.zeros_like(ff);qheat=0.
    # All directed physical events, both Pauli holes; no paired/expm1 code.
    for z in range(nc):
      F=A*ff[z,:,None]*(1-ff[z,None,:]);dn=F.sum(axis=0)-F.sum(axis=1)
      df[z]+=dn/W;qheat-=E@dn/nc
    ins=outs=ine=oute=0.
    for a in range(N):
      speed=abs(m[a])*nc/cross
      if m[a]>0:
        for z in range(nc):df[z,a]+=speed*((left[a]if z==0 else ff[z-1,a])-ff[z,a])
        incoming,outgoing=left[a],ff[-1,a]
      else:
        for z in range(nc):df[z,a]+=speed*((right[a]if z==nc-1 else ff[z+1,a])-ff[z,a])
        incoming,outgoing=right[a],ff[0,a]
      weight=W[a]*abs(m[a])/cross
      ins+=weight*incoming;outs+=weight*outgoing;ine+=weight*incoming*E[a];oute+=weight*outgoing*E[a]
    return df,np.array([ins,outs,ine,oute,qheat])

g=json.loads((HERE/'reviewed_initial_accepted_results.json').read_text())['grids'][-1]
E=np.array(g['E']);W=np.array(g['W']);m=np.array(g['mu']);A=np.array(g['A']);we=np.array(g['wE']);wa=np.array(g['wmu']);K=np.array(g['Kbar'])
ownW=we*E**2/(4*math.pi**2*197.3269804**3)*wa
ck('one-helicity polar phase measure',np.max(abs(W/ownW-1)),3e-15)
ck('outgoing polar measure no extra half or DOS',np.max(abs(A[~np.eye(len(E),dtype=bool)]/(ownW[:,None]*we[None,:]*wa[None,:]*K)[~np.eye(len(E),dtype=bool)]-1)),3e-15)
for en in np.unique(E):ck('polar weights sum to isotropic DOS at '+str(en),np.sum(W[E==en])/(we[E==en][0]*en**2/(2*math.pi**2*197.3269804**3))-1,3e-15)

edges=[]
for ep in [(7.3,11.2,.27,-.63),(8.4,8.4,-1/math.sqrt(3),1/math.sqrt(3)),(6.4,8.9,1/math.sqrt(3),1/math.sqrt(3))]:
    E0,F0,u,v=ep;ref=own_edge(*ep);tail=own_edge(*ep,tail=65)
    a=B.angular_edge(*ep,azimuth=128,inner=32,tail=80.);b=B.angular_edge(*ep,azimuth=256,inner=40,tail=90.)
    ck('fresh phase edge '+str(ep),b/ref-1,3e-6);ck('fresh edge own tail65/90 '+str(ep),tail/ref-1,3e-8);ck('fresh edge author128/256 '+str(ep),a/b-1,3e-6)
    reverse=own_edge(F0,E0,v,u)
    ck('fresh reverse DOS KMS '+str(ep),math.log(E0**2*ref/(F0**2*reverse))-(E0-F0)/2.,3e-10)
    edges.append(dict(point=ep,independent=ref,author128=a,author256=b,independent_reverse=reverse))
save('edge_results.json',dict(checks=checks,edges=edges,seconds=time.monotonic()-start))

# New anisotropic populations; independently ordered event and streaming sums.
f=np.array([expit((-.41-E)/(1.8+.3*m)),expit((-.21-E)/(2.3-.2*m))]);left=expit((-.41-E)/2.7);right=expit((-.41-E)/1.6);cross=.047
df,led=B.rhs(g,f,left,right,crossing_time=cross);dfi,li=independent_rhs(g,f,left,right,cross)
ck('fresh directed-event+mode streaming RHS',np.max(abs(df-dfi)),3e-12)
ck('fresh independently normalized boundary/heat ledger',np.max(abs(led[:5]-li))/np.max(abs(li)),3e-13)
for z in range(2):
 q=B.collision(g,f[z]);F=A*f[z,:,None]*(1-f[z,None,:]);dn=F.sum(0)-F.sum(1)
 ck('fresh number conservation '+str(z),sum(dn)/F.sum(),3e-14)
 ck('fresh bath heat sign '+str(z),(-E@dn-q['heat'])/max(abs(q['heat']),1e-300),3e-12)
 entropy=-np.log(f[z]/(1-f[z]))@dn+q['heat']/2
 ck('fresh thermostat entropy '+str(z),(entropy-q['entropy'])/F.sum(),3e-12)
 ck('fresh entropy nonnegative '+str(z),min(0,entropy)/F.sum(),1e-15)
for munu in [-1.13,.83]:
 occ=expit((munu-E)/2);c=B.collision(g,occ);ck('new LTE physical paired source '+str(munu),np.max(abs(c['dn']))/c['activity'],3e-12)

# Independent implicit Radau integration, actual holes re-evaluated every stage.
duration=.013;initial=np.tile(expit((-.41-E)/1.85),(2,1));n=initial.size
def fun(t,y):
 ff=y[:n].reshape(initial.shape);d,l=independent_rhs(g,ff,left,right,cross)
 return np.r_[d.ravel(),l]
sol=solve_ivp(fun,[0,duration],np.r_[initial.ravel(),np.zeros(5)],method='Radau',rtol=2e-10,atol=2e-18,max_step=.0005)
assert sol.success
end=sol.y[:n,-1].reshape(initial.shape);ledger=sol.y[n:,-1]
auth=B.evolve(g,cells=2,duration=duration,crossing_time=cross,left_T=2.7,right_T=1.6,rad_mu=-.41,initial_T=1.85,rtol=2e-11,max_step=.0005)
ck('new implicit Radau vs explicit DOP853 populations',np.max(abs(end-np.array(auth['final']))),3e-10)
ck('new independent ODE heat',ledger[4]/auth['thermostat_heat']-1,3e-7)
N=np.mean((end-initial)@W);U=np.mean((end-initial)@(W*E))
ck('independent ODE N/Le boundary ledger',(N-ledger[0]+ledger[1])/(ledger[0]+ledger[1]),3e-12)
ck('independent ODE E+bath+boundary reservoir ledger',(U+ledger[4]-ledger[2]+ledger[3])/(ledger[2]+ledger[3]+abs(ledger[4])),3e-12)
ck('independent ODE positivity',float(not np.all((sol.y[:n]>0)&(sol.y[:n]<1))),0.)
summary=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],edges=edges,independent_evolution=dict(final=end,ledger=ledger,N=N,U=U,nfev=sol.nfev),author_evolution=auth,seconds=time.monotonic()-start,initial_author_sha256=hashlib.sha256((HERE/'reviewed_initial_slab.py').read_bytes()).hexdigest(),scope='selected finite-grid physical edges, independent finite-slab ODE; no continuum convergence or repaired controls')
save('results.json',summary);print(json.dumps({k:summary[k]for k in ['passed','failed','seconds']}))
