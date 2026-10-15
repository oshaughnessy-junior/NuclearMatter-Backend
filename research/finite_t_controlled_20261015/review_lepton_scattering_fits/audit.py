"""Training-only QR reproduction and fresh actual detailed fallback checks."""
from pathlib import Path
import sys,json,math,hashlib,importlib.util,time
import numpy as np
from numpy.polynomial.chebyshev import chebvander,chebval
HERE=Path(__file__).resolve().parent;A=HERE.parent/'lepton_scattering_fits';sys.dont_write_bytecode=True
start=time.monotonic();checks=[]
def load(name,file):
 sp=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
API=load('independent_review_fit_api',A/'api.py');R=load('independent_review_fit_reference',A/'reference.py')
before={x.name:hashlib.sha256(x.read_bytes()).hexdigest()for x in A.iterdir()if x.is_file()}
model=json.loads((A/'model.json').read_text());reg=json.loads((A/'PREREGISTRATION.json').read_text());lock=json.loads((A/'FITTING_LOCK.json').read_text());train=json.loads((A/'training_references.json').read_text());hold=json.loads((A/'heldout_references.json').read_text());qual=json.loads((A/'QUALIFICATION.json').read_text());validation=json.loads((A/'validation_results.json').read_text())
def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(np.isfinite(error)and abs(error)<=tol)))
def test(name,passed):ck(name,0 if passed else 1,0)
def dump(name,data):(HERE/name).write_text(json.dumps(data,indent=2,default=lambda x:x.item()if isinstance(x,np.generic)else x.tolist())+'\n')
test('model hash locked',hashlib.sha256((A/'model.json').read_bytes()).hexdigest()==lock['model_sha']==API.MODEL_SHA)
test('registration hash locked',hashlib.sha256((A/'PREREGISTRATION.json').read_bytes()).hexdigest()==lock['preregistration_sha']==model['preregistration_sha'])
test('training data hash locked',hashlib.sha256((A/'training_references.json').read_bytes()).hexdigest()==lock['training_reference_sha']==model['training_reference_sha'])
test('recorded pre-holdout absence flag',lock['heldout_source_file_absent']is True)
test('actual training order declared',reg['training_MeV']==[x['E']for x in train['rows']])
test('actual heldout order declared',reg['heldout_MeV']==[x['E']for x in hold['rows']])
test('blocked holdout bands contain no training',all(not lo<=x<=hi for lo,hi in reg['heldout_blocks_MeV']for x in reg['training_MeV']))
test('all heldouts lie in their declared bands',all(lo<=x<=hi for x,(lo,hi)in zip(reg['heldout_MeV'],reg['heldout_blocks_MeV'])))
test('no source quadrature failures ignored',not train['failed']and not hold['failed'])

E=np.array(reg['training_MeV']);x=2*np.log(E)/np.log(30)-1
def qrfit(x,y,d):
 design=chebvander(x,d);q,r=np.linalg.qr(design,mode='reduced');return np.linalg.solve(r,q.T@y)
selections={}
for key in reg['positive_observables']:
 y=np.log([z['fine']['total'][key]for z in train['rows']]);candidates=[]
 for degree in reg['fit']['candidate_degrees']:
  c=qrfit(x,y,degree);tr=float(max(abs(np.expm1(chebval(x,c)-y))));errors=[]
  for j in range(len(x)):
   keep=np.arange(len(x))!=j;coeff=qrfit(x[keep],y[keep],degree);errors.append(float(np.expm1(chebval(x[j],coeff)-y[j])))
  loo=max(abs(e)for e in errors);candidates.append(dict(degree=degree,c=c,training=tr,loo=loo,passes=tr<=.003 and loo<=.007))
  saved=next(z for z in model['models'][key]['candidates']if z['degree']==degree)
  ck(key+' candidate coefficients '+str(degree),max(abs(c-np.array(saved['coefficients']))),3e-12)
  ck(key+' candidate LOO '+str(degree),loo-saved['LOO_max_relative'],3e-12)
 selected=next((z for z in candidates if z['passes']),min(candidates,key=lambda z:z['loo']));selections[key]=selected['degree']
 test(key+' independent training-only degree',selected['degree']==model['models'][key]['selected']['degree'])
test('twelve coefficients degrees3/4/2',list(selections.values())==[3,4,2]and sum(d+1 for d in selections.values())==12)
saved_fail=validation['failed'];test('exactly two original signed failures retained',len(saved_fail)==2 and [z['name']for z in saved_fail]==['training signedRT 19.0','training signedRT 30.0']and qual['signed_energy']['failed_checks']==saved_fail)
for E0 in [19.,30.]:
 row=next(z for z in train['rows']if z['E']==E0);ref=row['fine']['total'];fit=API.evaluate(E0)
 err=(fit['neutrino_energy_change']-ref['neutrino_energy_change'])/(ref['rate']*2.)
 expected=next(z['error']for z in saved_fail if z['name']=='training signedRT '+str(E0));ck('original signed failure reproduced '+str(E0),err-expected,2e-12)
 test('failed signed gate APIflag '+str(E0),fit['signed_energy_RT_gate_pass']is False)
test('positive qualification explicitly not uniform',qual['positive_moments']['uniform_error_certificate']is False)

bad=[('below_domain',lambda:API.evaluate(.999)),('above_domain',lambda:API.evaluate(30.001)),('NaN',lambda:API.evaluate(np.nan)),('Inf',lambda:API.evaluate(np.inf)),('bool_energy',lambda:API.evaluate(True)),('complex_energy',lambda:API.evaluate(8+0j)),('differentT',lambda:API.evaluate(8.,T=2.00001)),('NaNT',lambda:API.evaluate(8.,T=np.nan)),('differentmu',lambda:API.evaluate(8.,mu_e=R.MU+1e-12)),('antinu',lambda:API.evaluate(8.,flavor='anti_nu_e')),('one_target',lambda:API.evaluate(8.,targets=('e-',))),('FDclosure',lambda:API.evaluate(8.,outgoing='FD')),('method',lambda:API.evaluate(8.,method='cached'))]
for name,fn in bad:
 try:fn();ok=False
 except ValueError:ok=True
 test('API rejects '+name,ok)
rows=[]
for en in [8.19,23.17]:
 test('fresh point absent training/heldout '+str(en),en not in reg['training_MeV']+reg['heldout_MeV'])
 coarse=R.detailed(en,R.COARSE);fine=API.evaluate(en,method='detailed');fit=API.evaluate(en)
 for key in reg['positive_observables']:
  ck('fresh source coarse/fine '+str((en,key)),coarse['total'][key]/fine[key]-1,1e-4)
  ck('fresh positive fit '+str((en,key)),fit[key]/fine[key]-1,.01)
 ck('fresh source signedRT '+str(en),(coarse['total']['neutrino_energy_change']-fine['neutrino_energy_change'])/(fine['rate']*2),1e-4)
 ck('fresh signed diagnosticRT '+str(en),(fit['neutrino_energy_change']-fine['neutrino_energy_change'])/(fine['rate']*2),.03)
 test('fresh fallback actual method '+str(en),fine['method']=='actual detailed quadrature')
 rows.append(dict(E=en,coarse=coarse,fine_actual_fallback=fine,fit=fit))
 dump('partial_fresh_results.json',dict(checks=checks,rows=rows,seconds=time.monotonic()-start))
test('all author artifacts untouched',before=={z.name:hashlib.sha256(z.read_bytes()).hexdigest()for z in A.iterdir()if z.is_file()})
out=dict(checks=checks,passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],fresh=rows,training_only_degrees=selections,author_before=before,seconds=time.monotonic()-start,chronology_limit='hashes and source data dependencies corroborate locked recipe; absence of earlier private heldout knowledge is not cryptographically provable from files alone')
dump('results.json',out);print(json.dumps({k:out[k]for k in ['passed','failed','seconds']}))
