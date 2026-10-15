"""Retain coarse phi failures; qualify finer actual angular kernels/slab."""
from pathlib import Path
import sys,importlib.util,math,json,time
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
P=Path(__file__).parent;sys.dont_write_bytecode=True
def load(n,f):
 sp=importlib.util.spec_from_file_location(n,P/f);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
B=load('slab_refinement_owned','slab.py');R=load('slab_reference_owned','reference.py');old=json.loads((P/'results.json').read_text());checks=[];start=time.monotonic()
def ck(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def dump(n,z):(P/n).write_text(json.dumps(z,indent=2,default=lambda x:x.item()if isinstance(x,np.generic)else str(x)))
# Target populations forward before any derivatives or weak source conclusions.
pop=[]
for mu,ref in [(B.MU,B.LEPTON.particle_density),(-B.MU,B.LEPTON.antiparticle_density)]:
 n=quad(lambda p:p*p*expit((mu-math.hypot(p,B.S.ME))/B.T),0,250,epsabs=1e-11,epsrel=2e-12)[0]/(math.pi**2*B.HC**3);ck('actual pair population root '+str(mu),n/ref-1,2e-9);pop.append(n)
ck('actual net pair charge',(pop[0]-pop[1])/B.NE-1,2e-9)
grids=[]
for phi,inner,tail in[(128,32,80.),(256,40,90.)]:
 g=B.grid(azimuth=phi,inner=inner,tail=tail);grids.append(g);dump('refinement_partial.json',dict(grids=grids,checks=checks,pins=B.PINS))
a,b=grids;aa=np.array(a['Kbar']);bb=np.array(b['Kbar']);mask=bb>0;ck('all physical directional128to256 plusinner/tail',np.max(abs(aa[mask]/bb[mask]-1)),3e-6)
coarse=np.array(old['grids'][1]['Kbar']);scale=np.divide(abs(coarse-bb),bb,out=np.zeros_like(bb),where=mask);locations=np.dstack(np.unravel_index(np.argsort(scale.ravel())[::-1],scale.shape))[0];edges=[];seen=set()
for i,j in locations:
 if i==j or (min(i,j),max(i,j))in seen:continue
 seen.add((min(i,j),max(i,j)));E,F,u,v=[b[k][q]for k,q in [('E',i),('E',j),('mu',i),('mu',j)]];ref=R.edge(E,F,u,v,B.T,B.MU);ck('actual previously difficult phase edge '+str((int(i),int(j))),bb[i,j]/ref-1,3e-6);edges.append(dict(i=int(i),j=int(j),E=E,F=F,mu=u,mup=v,reference=ref,fine=bb[i,j]))
 if len(edges)==3:break
E=np.array(b['E']);W=np.array(b['W']);A=np.array(b['A']);f=expit((-.7-E)/B.T);q=B.collision(b,f);ck('final physical LTE collision',max(abs(q['dn']))/q['activity'],3e-12);ck('final physical reverse KMS',b['kms_residual'],3e-10)
ra=B.evolve(a,rtol=2e-11,max_step=.001);rb=B.evolve(b,rtol=2e-11,max_step=.001);ck('refined actual slab heat128to256',ra['thermostat_heat']/rb['thermostat_heat']-1,3e-6);ck('refined actual slab occupations',np.max(abs(np.array(ra['final'])-np.array(rb['final']))),3e-7)
# Independent predictor/corrector time rule against DOP853 on final physical grid.
left,right=B.boundary(E);f0=np.array(rb['initial']);nc=len(f0);duration=rb['duration'];runs=[]
for steps in[64,128,256]:
 ff=f0.copy();ledger=np.zeros(6);dt=duration/steps
 for n in range(steps):
  df,a0=B.rhs(b,ff,left,right);pred=ff+dt*df;df1,a1=B.rhs(b,pred,left,right);ff+=dt/2*(df+df1);ledger+=dt/2*(a0+a1)
 err=float(np.mean(abs(ff-np.array(rb['final']))@W)/np.mean(f0@W));runs.append(dict(steps=steps,final=ff.tolist(),ledgers=ledger.tolist(),weighted_error=err))
 ck('explicitHeun positivity '+str(steps),float(not np.all((ff>0)&(ff<1))),0);ck('explicitHeun number ledger '+str(steps),(np.mean((ff-f0)@W)-ledger[0]+ledger[1])/(ledger[0]+ledger[1]),3e-12);ck('explicitHeun energy ledger '+str(steps),(np.mean((ff-f0)@(W*E))+ledger[4]-ledger[2]+ledger[3])/(ledger[2]+ledger[3]+abs(ledger[4])),3e-12)
for a0,b0 in zip(runs[:-1],runs[1:]):ck('independentHeun observedorder '+str(a0['steps']),math.log2(a0['weighted_error']/b0['weighted_error'])-2,.15)
ck('independentHeun256 slabheat',runs[-1]['ledgers'][4]/rb['thermostat_heat']-1,.0005)
# Two finite volumes are a numerical ansatz: demonstrate spatial sensitivity, no convergence claim.
space=B.evolve(b,cells=6,rtol=2e-11,max_step=.001);spatial_heat=space['thermostat_heat']/rb['thermostat_heat']-1
r=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],preserved_initial_failed=old['failed'],actual_pair_densities=pop,grids=grids,edges=edges,coarse_run=ra,accepted_run=rb,Heun=runs,six_cell_run=space,spatial_heat_change=spatial_heat,pins=B.PINS,seconds=time.monotonic()-start,scope='finiteenergy24MeV S2 axisymmetric threecellupwind slab; fixedpairthermostat; no continuum in angle/space/energy')
dump('accepted_results.json',r);print(json.dumps(dict(passed=r['passed'],failed=r['failed'],seconds=r['seconds'],heat=rb['thermostat_heat'],spatial_heat_change=spatial_heat),indent=2))
