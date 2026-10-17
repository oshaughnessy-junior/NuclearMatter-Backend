from pathlib import Path
import sys,json,math,dataclasses,time,hashlib
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'candidate_copy/src'))
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _weak_current as C
L=json.loads((P/'LOCK.json').read_text());checks=[];rows=[];ch=sys.argv[1];assert ch in['EC','PC','NEUTRON_DECAY'];start=time.monotonic();out=dict(channel=ch,checks=checks,rows=rows,stage='constants/forwardreference',complete=False)
def ck(n,e,g=0):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
def save():
 out.update(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],seconds=time.monotonic()-start);(P/f'candidate_{ch}.json').write_text(json.dumps(out,indent=2)+'\n')
def admission():
 save()
 if any(not c['passed']for c in checks):raise ArithmeticError('Forwardcandidatefindingpreserved')
# Exactphysicalidentity and previouslyindependentgiven-muFDnumbergates first.
for n,v in L['masses'].items():ck('exactphysicalconstant '+n,0 if getattr(C,n)==v else 1)
pt=json.loads((P/'point_results.json').read_text());assert pt['stage']=='COMPLETE'and not pt['failed'];out['independent_forward_densities']=pt['forwarddensities'];ck('actualFDdensitybeforecandidate',0 if all(abs(d['original_p']/d['saved']-1)<3e-9 for d in pt['forwarddensities'].values())else 1)
b=W.FreeElectronBath(**L['bath']);ctrl=W.SpectralControls.fine();admission();out['stage']='differential'
for p in pt['rows']:
 if p['input']['channel']!=ch:continue
 v=p['input'];r=W.capture_differential_coefficients(ch,b,v['E'],v['El'],v['cosine'],ctrl);row=dataclasses.asdict(r);rows.append(dict(kind='heldoutdifferential',input=v,candidate=row));ref=p['delta'][-1]
 for key,field in[('j','j'),('chi','true_chi')]:
  z=getattr(r,field);ck('independentdiff '+v['label']+key,math.expm1(z.log-ref['log'+key]),3e-10);ck('positivefinite '+key+v['label'],0 if z.support=='positive_support'and z.value>0 else 1)
 admission()
out['stage']='integrated'
selected=json.loads((P/'ANALYSIS.json').read_text())['selected_references']
for v in selected:
 if v['channel']!=ch:continue
 r=W.spectral_coefficients(ch,b,v['E'],ctrl);rr=dataclasses.asdict(r);rows.append(dict(kind='heldoutintegrated',E=v['E'],reference=v['qualified_reference_file'],candidate=rr))
 for key,field in[('j','j'),('chi','true_chi')]:
  z=getattr(r,field);ck('integratedpositive '+key+str(v['E']),0 if z.support=='positive_support'and math.isfinite(z.log)and(z.value is None or z.value>0)else 1);ck('independentintegrated '+key+str(v['E']),math.expm1(z.log-v['log'+key]),3e-6)
 admission()
 # Posterioridentities only AFTER bothbranches absolutecheck.
 ck('relaxation '+str(v['E']),r.relaxation.value/(r.j.value+r.true_chi.value)-1,3e-15);ck('KMSposterior '+str(v['E']),r.kms_log_residual,3e-10);ck('lengthcm '+str(v['E']),r.absorption_cm_inverse.value/(r.true_chi.value/2.99792458e10)-1,5e-15);ck('lengthkm '+str(v['E']),r.absorption_km_inverse.value/(r.true_chi.value/2.99792458e5)-1,5e-15);ck('trueabsorptiondiffersrelax '+str(v['E']),0 if r.relaxation.value>r.true_chi.value else 1);admission()
for n,v in json.loads((P/'CANDIDATE_PIN.json').read_text())['candidate_files'].items():assert hashlib.sha256((P/'candidate_copy'/n).read_bytes()).hexdigest()==v,n
out.update(stage='COMPLETE',complete=True,origin=W.__file__);save();print(ch,'PASS',out['passed'],'FAIL',len(out['failed']),'seconds',out['seconds'])
