from pathlib import Path
import sys,json,math,dataclasses,hashlib,time,ast
import numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'candidate_copy/src'))
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_spectral_common as S,_weak_spectral_capture as K
from nmbackend.reference import _weak_quadrature as Q
L=json.loads((P/'LOCK.json').read_text());b=W.FreeElectronBath(**L['bath']);c=W.SpectralControls.fine();checks=[];witnesses=[];pilots=[];exceptions=[];start=time.monotonic()
def ck(n,e=0,g=0):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
def reject(n,fn,typ=ValueError):
 try:z=fn()
 except typ as e:ck(n);exceptions.append(dict(name=n,type=type(e).__name__,message=str(e)))
 except Exception as e:ck(n+'wrongexception',1);exceptions.append(dict(name=n,type=type(e).__name__,message=str(e)))
 else:
  ck(n,1);witnesses.append(dict(name=n,accepted_type=type(z).__name__,accepted=dataclasses.asdict(z)if dataclasses.is_dataclass(z)else repr(z)))
def save():
 out=dict(passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],checks=checks,witnesses=witnesses,pilots=pilots,exception_evidence=exceptions,seconds=time.monotonic()-start,scope='Physics/localadmissionguardreview only; no wholeRQ/continuumqualification');(P/'GUARD_RESULTS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
# Positivenodephysics already independentlyqualified BEFOREusingrecordsbelow.
for ch in['EC','PC','NEUTRON_DECAY']:
 f=json.loads((P/f'candidate_{ch}.json').read_text());assert f['complete']and not f['failed']
# Actual finite-range pilot, directFDpositivebranches and independent same-nodeDOSsum.
rad=W.Radiation.fd(b.T,b.equilibrium_mu_nue-1.4,b.equilibrium_mu_antinue+1.8)
for ch,lo,hi in[('EC',5.8,6.8),('PC',3.7,4.7),('NEUTRON_DECAY',.82,.92)]:
 rule=W.EnergyRule.gauss(lo,hi,2);r=W.reconstruct_spectral_moments(ch,b,rad,rule,c);pilots.append(dataclasses.asdict(r));save();F=[0.,0.];B=[0.,0.];entropy=0.
 for E,w,n in zip(rule.nodes,rule.weights,r.nodes):
  mu=rad.mu_nue if ch=='EC'else rad.mu_antinue;f=1/(1+math.exp((E-mu)/b.T));jj=n.j.value*(1-f);xx=n.true_chi.value*f;weight=w*E*E/(2*math.pi**2*S.C.HC**3);entropy+=weight*(jj-xx)*math.log(jj/xx)
  for i in[0,1]:F[i]+=weight*E**i*jj;B[i]+=weight*E**i*xx
 for d,v in[('forward',F),('backward',B)]:
  for i,x in enumerate(v):ck(ch+d+str(i)+' singleDOS+holeonce',getattr(r,d)[i].value/x-1,3e-14)
 for i in[0,1]:ck(ch+'directnet'+str(i),(r.net[i].value-F[i]+B[i])/(F[i]+B[i]),3e-9)
 ck(ch+'directentropy',(r.entropy_production.value-entropy)/(F[0]+B[0]),3e-10);ck(ch+'finiteonlyscope',0 if r.complete_finite_range and r.whole_moment_qualified is False else 1)
 for name,kwargs in[('wholeclaim',dict(whole_moment_qualified=True)),('partial',dict(complete_finite_range=False)),('missingnode',dict(nodes=r.nodes[:-1])),('swappednode',dict(nodes=r.nodes[::-1])),('wrongunits',dict(units=('s^-1','MeVs^-1'))),('changedcontrol',dict(controls=W.SpectralControls()))]:reject(ch+name,lambda kwargs=kwargs:dataclasses.replace(r,**kwargs))
 if ch=='EC':
  # Narrow admissioncounterexamples: freshrealnodes, butmetadata/aggregatechanged.
  eqrad=W.Radiation.fd(b.T,b.equilibrium_mu_nue,b.equilibrium_mu_antinue)
  reject('REQUIRED FDmetadata changed while staleaggregates retained',lambda:dataclasses.replace(r,radiation=eqrad))
  reject('REQUIRED positiveaggregate changed while actualnodes retained',lambda:dataclasses.replace(r,forward=(Q.positive(r.forward[0].log+math.log(2)),r.forward[1])))
  altered=[]
  for E,n in zip(rule.nodes,r.nodes):
   f=1/(1+math.exp((E-rad.mu_nue)/b.T));delta=.2*min(n.j.value*(1-f),n.true_chi.value*f);nj=n.j.value+delta/(1-f);nx=n.true_chi.value+delta/f;altered.append(dataclasses.replace(n,j=S.coefficient(math.log(nj)),true_chi=S.coefficient(math.log(nx))))
  # EachnodeFandBchangeequally, preservingnetR/Q butNOTpositivebranchmoments.
  reject('REQUIRED changedBOTHbranches preservingnet but stalepositiveF/B',lambda:dataclasses.replace(r,nodes=tuple(altered)))
 save()
# Positive differential forwardpoint; changingonlyfakeemptywitnessmustreject.
good=W.capture_differential_coefficients('EC',b,4.61,5.897,.27,c);ck('freshpositiveheldout',0 if good.j.value>0 and good.true_chi.value>0 else 1)
reject('REQUIRED exactempty forged at provedpositivepoint',lambda:dataclasses.replace(good,j=S.empty_coefficient(),true_chi=S.empty_coefficient(),geometry=('analytic_empty_particle_hole_support',)))
empty=W.capture_differential_coefficients('PC',b,4.61,5.897,.99987,c);ck('actualanalyticPCempty',0 if empty.j.log is None and empty.true_chi.log is None and empty.j.value==empty.true_chi.value==0.else 1)
# Legacyoptin, strictstate/types/controls/representation andactualtimeout.
for ch in['EC','PC','NEUTRON_DECAY']:reject('legacyNone '+ch,lambda ch=ch:W.spectral_coefficients(ch,b,.5),NotImplementedError)
for x in[True,np.bool_(False),float('nan'),float('inf'),10**10000,0.,-1.,61.]:reject('energy'+str(type(x))+str(len(checks)),lambda x=x:W.spectral_coefficients('EC',b,x,c))
for bath in[None,object(),dict(L['bath']),dataclasses.replace(b,mu_e=math.nextafter(b.mu_e,math.inf))]:reject('exactbath'+str(type(bath)),lambda bath=bath:W.spectral_coefficients('EC',bath,5.,c))
for name in['lepton_order','angle_order','energy_order','phi_order','parent_order','cone_order','electron_order','decay_phi_order']:
 for x in[True,4.,3,33,10**10000]:reject(name+str(type(x))+str(len(checks)),lambda name=name,x=x:W.SpectralControls(**{name:x}))
for kw in[dict(tail=True),dict(tail=29.9),dict(tail=100.1),dict(outer_tail=float('nan')),dict(outer_tail=19.9),dict(outer_tail=60.1),dict(budget_seconds=False),dict(budget_seconds=0.),dict(budget_seconds=float('inf')),dict(budget_seconds=120.1)]:reject('controls'+str(kw),lambda kw=kw:W.SpectralControls(**kw))
for args in[(0.,-1000.,'positive_support','underflow'),(0.,None,'positive_support','represented'),(True,0.,'positive_support','represented'),(None,float('-inf'),'positive_support','underflow'),(0.,None,'unresolved','x'),(False,None,'exact_kinematic_zero','analytic_empty_particle_hole_support')]:reject('coefficientrecord'+str(args),lambda args=args:W.SpectralQuantity(*args))
for log in[-1000.,-13836.,1000.]:
 z=S.coefficient(log);ck('finite-lognoordinaryzero '+str(log),0 if z.log==log and z.value is None and z.support=='positive_support'else 1)
for ch in['EC','PC','NEUTRON_DECAY']:
 reject('timeoutactual '+ch,lambda ch=ch:W.spectral_coefficients(ch,b,.5,dataclasses.replace(c,budget_seconds=1e-12)),TimeoutError)
reject('reconFDmalformedstate',lambda:W.reconstruct_spectral_moments('EC',object(),rad,W.EnergyRule.gauss(2.,3.,2),c));reject('reconFDwrongT',lambda:W.reconstruct_spectral_moments('EC',b,W.Radiation.fd(1.9,0.,0.),W.EnergyRule.gauss(2.,3.,2),c))
reject('reconactualsharedtimeout',lambda:W.reconstruct_spectral_moments('EC',b,rad,W.EnergyRule.gauss(2.,3.,2),dataclasses.replace(c,budget_seconds=1e-12)),TimeoutError)
reject('geometricqzeroNOTempty',lambda:K.support(S.C.MN,S.C.MP,-.5,0.),ArithmeticError);reject('degeneratesupportNOTfalseempty',lambda:K.support(2.,1.,1.25,.75),ArithmeticError)
for x in[S.C.MN,S.C.MP]:ck('positivephysicalmass',0 if x>0 else 1)
# Runtimeorigins/staticimports andimmutablefiles.
allow={'__future__','dataclasses','math','numpy','scipy'}
for f in(P/'candidate_copy/src/nmbackend/reference').glob('_weak_spectral*.py'):
 tree=ast.parse(f.read_text());bad=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Import):bad.extend(x.name for x in n.names if x.name.split('.')[0]not in allow)
  if isinstance(n,ast.ImportFrom)and n.level==0 and n.module.split('.')[0]not in allow:bad.append(n.module)
 ck('portableimports'+f.name,len(bad));ck('nofileIO'+f.name,0 if all(k not in f.read_text()for k in['importlib','sys.path','read_text(','outputs/','loadtxt('])else 1);ast.parse(f.read_text(),feature_version=(3,10));ck('minimumsyntaxonly'+f.name)
for n,v in json.loads((P/'CANDIDATE_PIN.json').read_text())['candidate_files'].items():ck('candidatecopyunchanged '+n,0 if hashlib.sha256((P/'candidate_copy'/n).read_bytes()).hexdigest()==v else 1)
save();print('GUARDS',sum(z['passed']for z in checks),'FAIL',sum(not z['passed']for z in checks),'seconds',time.monotonic()-start);print([z['name']for z in checks if not z['passed']])
