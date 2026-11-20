"""Independent antinue-spectator Dalitz and DIRECT parent-momentum review."""
from pathlib import Path
import sys, math, json, time, hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit, logsumexp
from scipy.integrate import quad
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1]; H=Path(__file__).parent
sys.path.insert(0,str(R/'muon_thermal_events'))
import muon_events as author
M=105.6583755; me=.51099895069
GF=author.vacuum.GF; hbar=author.vacuum.HBAR; hc=author.HBARC

def independent(b,rad,n=24):
    start=time.monotonic(); x,w=leggauss(n)
    # Single out anti-nu_e, rather than the author's electron spectator.
    th=(x+1)*math.pi/4; Amax=(M*M-me*me)/(2*M)
    av=Amax*np.sin(th)**2; wa=w*math.pi/4*2*Amax*np.sin(th)*np.cos(th)
    aa,c=np.meshgrid(av,x,indexing='ij'); aa=aa.ravel(); c=c.ravel()
    t=M*M-2*M*aa
    mid=(M-aa)*(t+me*me)/(2*t); half=aa*(t-me*me)/(2*t)
    ee=mid+half*c; nn=M-ee-aa; pe=np.sqrt((ee-me)*(ee+me))
    # n^2=0 fixes opening angle; separate direct scalar amplitude.
    ca=(ee*aa-M*(ee+aa)+(M*M+me*me)/2)/(pe*aa)
    assert np.max(abs(ca))<1+1e-10
    amp=32*GF*GF*M*aa*(t-me*me)
    rw=(wa[:,None]*w[None,:]).ravel()*half*amp/(64*math.pi**3*M*hbar)
    u,wu=leggauss(n); ph=2*math.pi*np.arange(2*n)/(2*n)
    U,P=np.meshgrid(u,ph,indexing='ij'); U=U.ravel(); P=P.ravel()
    oz=np.repeat(wu/2/(2*n),2*n)
    az=aa[:,None]*U
    ez=pe[:,None]*(ca[:,None]*U+np.sqrt(1-ca*ca)[:,None]*np.sqrt(1-U*U)*np.cos(P))
    nz=-az-ez
    base=(rw[:,None]*oz).ravel(); lb=np.log(base)
    ee0=np.broadcast_to(ee[:,None],ez.shape).ravel()
    aa0=np.broadcast_to(aa[:,None],az.shape).ravel()
    nn0=np.broadcast_to(nn[:,None],nz.shape).ravel()
    ez=ez.ravel(); az=az.ravel(); nz=nz.ravel()
    ku=max(0,b.mu_mu-M,b.mu_e+rad.mu_anti_e+rad.mu_nu_mu-M)+60*b.T
    pc=lambda k:math.sqrt(k*(k+2*M))
    cuts=sorted({0.,pc(ku)}|{pc(k)for z in [-20,-8,-3,0,3,8,20]if 0<(k:=b.mu_mu-M+z*b.T)<ku})
    F=[[]for _ in range(5)]; B=[[]for _ in range(5)]; nets=np.zeros(5)
    event_error=0.; ratio_error=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
      for xx,ww in zip(x,w):
        if time.monotonic()-start>115:raise TimeoutError('Independent bounded experiment')
        p=lo+(xx+1)*(hi-lo)/2; E=math.hypot(M,p); g=E/M; beta=p/E
        e=g*(ee0+beta*ez); a=g*(aa0+beta*az); nu=g*(nn0+beta*nz)
        pw=p*p/(math.pi**2*hc**3)*(M/E)*ww*(hi-lo)/2
        f=log_expit((b.mu_mu-E)/b.T)+log_expit((e-b.mu_e)/b.T)+log_expit((a-rad.mu_anti_e)/b.T)+log_expit((nu-rad.mu_nu_mu)/b.T)
        z=log_expit((E-b.mu_mu)/b.T)+log_expit((b.mu_e-e)/b.T)+log_expit((rad.mu_anti_e-a)/b.T)+log_expit((rad.mu_nu_mu-nu)/b.T)
        affinity=(b.mu_mu-b.mu_e-rad.mu_anti_e-rad.mu_nu_mu)/b.T
        ratio_error=max(ratio_error,float(np.max(abs(f-z-affinity))))
        event_error=max(event_error,float(np.max(abs(e+a+nu-E))/E))
        signed=np.exp(np.maximum(f,z))*np.where(f>=z,-np.expm1(-abs(f-z)),np.expm1(-abs(f-z)))
        for i,v in enumerate([np.ones_like(e),e,a,nu,np.full_like(e,E)]):
          F[i].append(float(logsumexp(lb+math.log(pw)+f+np.log(v))))
          B[i].append(float(logsumexp(lb+math.log(pw)+z+np.log(v))))
          nets[i]+=float(np.sum(base*pw*signed*v))
    return dict(forward=np.exp([logsumexp(v)for v in F]).tolist(),backward=np.exp([logsumexp(v)for v in B]).tolist(),net=nets.tolist(),event_error=event_error,local_affinity_error=ratio_error,runtime=time.monotonic()-start,order=n)

if __name__=='__main__':
    checks=[]; rows=[]
    def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(abs(error)<=tol and math.isfinite(error))))
    def save():
      (H/'results.json').write_text(json.dumps(dict(checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
    matter=author.eos.matter_state(3.,8e-5,.25,.015)
    b=author.Bath(3.,matter.muon.chemical_potential,matter.electron.chemical_potential)
    rad=author.Radiation(3.,8.,b.mu_mu-b.mu_e-8.-.4*b.T)
    rows.append(dict(state=dict(T=3.,nB=8e-5,Ye=.25,Ymu=.015),bath=author.asdict(b),radiation=author.asdict(rad)))
    # Independent direct-p FD density, including negative muon diagnostic.
    den=lambda mu:quad(lambda p:p*p/(math.pi**2*hc**3)*float(np.exp(log_expit((mu-math.hypot(M,p))/b.T))),0,700,epsabs=1e-19,epsrel=1e-11)[0]
    gate('fresh physical muon density',den(b.mu_mu)/(8e-5*.015)-1,2e-10)
    rows.append(dict(independent_muon_density=den(b.mu_mu),independent_antimuon_density=den(-b.mu_mu)))
    for order in [24,32]:
      r=independent(b,rad,order);rows.append(dict(independent=r));save()
    target=author.integrate(b,rad,parent_order=32,pair_order=40,pair_angle=24,orientation_order=20,azimuth_order=32,tail=60.)
    rows.append(dict(author_fresh=target));save()
    r1=rows[2]['independent'];r2=rows[3]['independent']
    keys=['number','electron_energy','antinue_energy','numu_energy','parent_energy']
    for branch in ['forward','backward','net']:
      for i,key in enumerate(keys):
        a=target[branch][key] if branch=='net' else target[branch][key]['value']
        gate('independent refinement '+branch+' '+key,r1[branch][i]/r2[branch][i]-1,3e-6)
        gate('absolute author comparison '+branch+' '+key,a/r2[branch][i]-1,3e-6)
    gate('independent event energy',r2['event_error'],2e-14)
    gate('local product chemical affinity',r2['local_affinity_error'],3e-12)
    net=r2['net']; q=net[2]+net[3]; rate=net[0]
    gate('parent-electron-radiation independent energy',(net[4]-net[1]-q)/q,3e-13)
    sm=(-q+(b.mu_mu-b.mu_e)*rate)/b.T
    sr=(q-(rad.mu_anti_e+rad.mu_nu_mu)*rate)/b.T
    gate('independent entropy affinity',(sm+sr)/(.4*rate)-1,3e-13)
    gate('author matter heat',target['ledgers']['material_energy']/(-q)-1,3e-6)
    gate('author total entropy',target['entropy']['total']/(sm+sr)-1,3e-6)
    gate('positive entropy sign',min(sm+sr,0),0)
    save();print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']],rows=len(rows))))
