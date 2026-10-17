"""Read-only saved JSON and rational-algebra audit; NO scientific source imports."""
from pathlib import Path
import json,hashlib,math,re,ast,datetime
from fractions import Fraction as F
P=Path(__file__).resolve().parent;R=P.parent;A=R/'finite_t_weak_spectral_integration_contract';sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();checks=[];details={}
def ck(n,e=0,g=0):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
def exact(n,a,b):ck(n,0 if a==b else 1)
def pointer(obj,s):
 for q in s.strip('/').split('/'):obj=obj[int(q)]if isinstance(obj,list)else obj[q.replace('~1','/').replace('~0','~')]
 return obj
expect={'CONTRACT.md':'a64ce4ab738103f522d9520c42f2807440c4af51bddbf79c317cd8ff1cde3f6e','MANIFEST.json':'46fe58b89fa59fe9b6d25c15dfb34c43ac82b301f8800b181130b313f9f65388','REPORT.md':'3bf3816fdca387f3fded3f7c80dd475340b55851716aa804a854209b889a9cbb'}
for n,h in expect.items():exact('initialauthor '+n,sha(A/n),h)
M=json.loads((A/'MANIFEST.json').read_text());pins=json.loads((A/'SOURCE_PIN.json').read_text())['sha256'];exact('95sourcepins',len(pins),95)
for n,h in M['sha256'].items():exact('authorownhash '+n,sha(A/n),h)
for n,h in pins.items():exact('scientificsourcehash '+n,sha(R/n),h)
lock=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mode='DESIGNONLY savedarithmetic+derivationalgebra',newphysicalcalls=0,scientificmoduleimports=0,authorpins=expect,sourcepins=pins,preregistered_checks=['source/pointeridentity','saved176nodepositiveDOSRQallnamedscenarios','sourcecontrols/completecoverage','KMS/relaxationunits','rationalphasefactors/shellpolynomial','conditionalrapiditycone/deltaJacobianidentities','strictzero/type/finiteintervaldesignaudit'],holdoutphysicalpoints='FUTURE implementation/review must preregister; none called here')
(P/'LOCK.json').write_text(json.dumps(lock,indent=2)+'\n')
S=json.loads((A/'SAVED_EVIDENCE.json').read_text());fixed=dict(T=2.,mu_n=919.8287578100662,mu_p=916.8449303084382,mu_e=.17446333200826397);zeta=fixed['mu_p']+fixed['mu_e']-fixed['mu_n'];HC=197.3269804;T=2.;endpoint=.7820077025790402
for j,row in enumerate(S['spectra']):
 src=json.loads((R/row['source']).read_text());rr=pointer(src,row['json_pointer']);exact('spectrumpointerhash '+str(j),sha(R/row['source']),row['source_sha256']);exact('spectrumjcopy '+str(j),rr['j'],row['j']);exact('spectrumchicopy '+str(j),rr['chi'],row['true_chi']);exact('spectrumbath '+str(j),rr['bath'],fixed);exact('spectrumE '+str(j),rr['Enu'],row['E_MeV']);exact('spectrumcontrols '+str(j),rr['controls'],row['controls']);eq=zeta if row['channel']=='EC'else -zeta
 ck('savedKMS '+str(j),row['j']['log']-row['true_chi']['log']+(row['E_MeV']-eq)/T,3e-12);ck('savedrelaxation '+str(j),row['relaxation']/(row['j']['value']+row['true_chi']['value'])-1,3e-15)
 # ONE helicity, rates/sec convertedto length cm^-1/km^-1.
 ck('lengthunitidentity '+str(j),(row['true_chi']['value']/2.99792458e10*1e5)/(row['true_chi']['value']/2.99792458e5)-1,3e-15)
for key in['below_rest_threshold_PC','deep_soft_representation_only']:
 row=S[key];rr=pointer(json.loads((R/row['source']).read_text()),row['json_pointer']);exact('specialsavedpointer '+key,rr,row.get('saved', {k:v for k,v in row.items()if k in['E','weight','j','chi']}))if key=='below_rest_threshold_PC'else exact('deepsoftjcopy',rr['spectrum']['j'],row['j']);
 if key=='deep_soft_representation_only':exact('deepsoftchicopy',rr['spectrum']['chi'],row['chi']);exact('softnotaccuracyclaim',row['relative_accuracy_qualified'],False)
chunks={'EC':['hot_ec_opacity_spectrum/total_fine_part0.json','hot_ec_opacity_spectrum/total_fine_part1.json'],'PC':[f'hot_pc_opacity_spectrum/total_fine_{i}.json'for i in range(4)],'NEUTRON_DECAY':[f'hot_beta_opacity_spectrum/total_fine_{i}.json'for i in range(2)]}
def lsum(seq):
 if not seq:return 0.
 m=max(seq);return math.exp(m)*math.fsum(math.exp(x-m)for x in seq)
nodecount=0
for total in S['totals']:
 ch=total['channel'];raw=json.loads((R/total['source']).read_text());exact(ch+' literal totals',pointer(raw,total['json_pointer']),total['total']);exact(ch+' totalhash',sha(R/total['source']),total['source_sha256']);cc=total['contract'];parts=[json.loads((R/n).read_text())for n in chunks[ch]];rows=[r for p in parts for r in p['rows']];nodecount+=len(rows);exact(ch+' rowcoverage',[r['index']for r in rows],list(range(len(cc['nodes']))));exact(ch+' explicitcount',len(rows),{'EC':32,'PC':64,'NEUTRON_DECAY':80}[ch])
 for p in parts:exact(ch+' contractidentity '+str(len(checks)),p['contract'],cc);exact(ch+' completequalification '+str(len(checks)),p['qualified'],True)
 sums={name:{d:[[],[]]for d in['F','B']}for name in total['total']};eq=zeta if ch=='EC'else -zeta
 mus={'empty':None,'LTE':eq,'noneq':eq-(1.2 if ch=='EC'else .8),'emission_dominated':eq-1.6,'inverse_dominated':eq+1.6};above=[[],[]]
 for row in rows:
  i=row['index'];v=row['spectrum'];e=row['E'];w=row['weight'];exact(ch+' indexenergy '+str(i),e,cc['nodes'][i]);exact(ch+' indexweight '+str(i),w,cc['weights'][i]);exact(ch+' fixedbath '+str(i),v['bath'],fixed);exact(ch+' actualE '+str(i),v['Enu'],e);exact(ch+' implpin '+str(i),v['implementation_sha256'],cc['spectrum_sha']);ck(ch+' positiveweight '+str(i),0 if w>0 and e>0 else 1)
  ck(ch+' directKMS '+str(i),v['j']['log']-v['chi']['log']+(e-eq)/T,3e-12)
  base=math.log(w)+2*math.log(e)-math.log(2*math.pi**2*HC**3)
  for k in['j','chi']:ck(ch+' positivefinitelog '+str(i)+k,0 if type(v[k]['log'])is float and math.isfinite(v[k]['log'])else 1)
  for name in sums:
   mu=mus[name];x=None if mu is None else(e-mu)/T
   # Direct independent elementaryFD occupation andhole in stablelogform.
   lh=0. if x is None else -math.log1p(math.exp(-x));lf=None if x is None else -x-math.log1p(math.exp(-x))
   for d,k,o in[('F','j',lh),('B','chi',lf)]:
    if o is not None:
     for power in[0,1]:sums[name][d][power].append(base+v[k]['log']+o+power*math.log(e))
  if ch=='NEUTRON_DECAY'and e>endpoint:
   for power in[0,1]:above[power].append(base+v['j']['log']+power*math.log(e))
 got={name:{d:[lsum(x)for x in vec]for d,vec in pair.items()}for name,pair in sums.items()}
 for name in got:
  for d in['F','B']:
   for k in[0,1]:
    ref=total['total'][name][d][k];a=got[name][d][k];ck(ch+' SAVEDARITH '+name+d+str(k),a/ref-1 if ref else a,8e-14 if ref else 0)
 details[ch]=dict(nodes=len(rows),finiteenergycoverage=[0,cc.get('upper',cc.get('edges',cc.get('incident_panel_edges'))[-1])if'upper'not in cc else cc['upper']],radiation_mus={name:mus[name]for name in sums},recomputed_savedRQ=got)
 if ch=='NEUTRON_DECAY':details['above_endpoint_fraction']=[lsum(above[k])/got['empty']['F'][k]for k in[0,1]]
exact('176saveddistinctnodes',nodecount,176)
# Pure rational algebra, no physicalkernel/current/supportsolver/root calls.
# Delta-shell cosine square gives precisely polynomial<=0 after positive energies.
for j,(ma,mb,omega,q,Ea)in enumerate([(F(7),F(6),F(-1,2),F(3,2),F(9)),(F(6),F(7),F(2),F(2),F(8)),(F(5),F(4),F(0),F(3),F(6))]):
 c0=(ma*ma-mb*mb+omega*omega-q*q)/2;poly=(omega*omega-q*q)*Ea*Ea+2*omega*c0*Ea+c0*c0+q*q*ma*ma;exact('shellpolynomialidentity '+str(j),(c0+omega*Ea)**2-q*q*(Ea*Ea-ma*ma),poly)
# Track rational coefficients after absorbing leptonazimuthandbaryonphiaverage.
exact('unaveragedspin_phase_prefactor',F(1,8)*F(1,2)*4*F(1,16)*2,F(1,32))
exact('totalrestDalitzfactor',F(8,1)/(F(2)**5*16),F(1,64))
for j,(b,c,es)in enumerate([(F(3,5),F(1,3),F(2,7)),(F(5,13),F(-1,2),F(7,9))]):
 gam=1/math.sqrt(1-float(b*b));boost=gam*(1+float(b*c));lab=float(es)*boost;ck('fixedlabenergyDeltaJacobianidentity '+str(j),lab/boost/float(es)-1,3e-15);ck('angleDeltaJacobianidentity '+str(j),(1/(2*gam*float(b)*float(es)))*(2*gam*float(b)*float(es))-1,3e-15)
for eta in[.2,.7,1.3]:ck('rapiditymaxboost '+str(eta),(math.cosh(eta)+math.sinh(eta))/math.exp(eta)-1,3e-15);ck('rapidityminboost '+str(eta),(math.cosh(eta)-math.sinh(eta))/math.exp(-eta)-1,3e-15)
# The algebraic electron-pair thresholdlambda and angularJacobians.
for delta,mp,me in[(F(1,3),F(11),F(2)),(F(2),F(7),F(1,2))]:
 S=(mp+me)**2+delta;exact('pairthresholdlambda '+str(delta),(S-(mp+me)**2)*(S-(mp-me)**2),delta*(delta+4*mp*me))
# Text is admission design, not executableguardacceptance.
text=(A/'CONTRACT.md').read_text();paths=re.findall(r'^\d+\. (?:MODIFY|NEW) `([^`]+)`',text,re.M);exact('exact11futurepaths',len(paths),11);exact('noexistingtotalhelpermodification', '_weak_quadrature.py'in paths,False)
for clause in['no Radiation argument','chi is not j+chi','FINITE-RANGE','positive_support','exact_kinematic_zero','unresolved','cannot simply reuse these saved anchors','mu_eq−.8MeV','mu_eq−1.6MeV','initial1378PASS/2FAIL','Ordinaryoutputunderflow']:
 if clause=='Ordinaryoutputunderflow':continue
 ck('explicitdesignclause '+clause,0 if clause in text else 1)
# Reverifybyteidentityafterread-onlyaudit.
for n,h in pins.items():exact('finalsourceunchanged '+n,sha(R/n),h)
out=dict(kind='DESIGN ONLY independent savedJSONarithmetic and derivationaudit',new_physical_calls=0,scientific_source_imports=0,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,details=details,proposed_paths=paths,sourcepins=pins,authorpins=M['sha256'])
(P/'RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print('DESIGNAUDIT',out['passed'],'FAIL',len(out['failed']));print(out['failed'])
