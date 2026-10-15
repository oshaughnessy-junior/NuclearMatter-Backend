"""Check frozen numerical evidence without redoing the expensive integrals."""
from pathlib import Path
import json,math,hashlib,platform
import numpy as np,scipy
P=Path(__file__).resolve().parent;ROOT=P.parent
checks=[]
def ck(name,error,tol):checks.append(dict(name=name,error=abs(float(error)),limit=tol,passed=bool(abs(error)<=tol)))
mu=json.loads((P/'mu_edge_refinement_results.json').read_text());q16=json.loads((P/'mu_q16.json').read_text());q20=json.loads((P/'mu_q20.json').read_text());controls=json.loads((P/'controls_results.json').read_text());anchor=json.loads((P/'zero_field_results.json').read_text())
q24=json.loads((P/'mu_q24.json').read_text());cos20=json.loads((P/'mu_cos20.json').read_text())
e8=next(r for r in controls if r['probe']=='electron8');ef=next(r for r in controls if r['probe']=='electron12q')
for r in mu+[q16,q20,q24,cos20]+controls+anchor:
 ck('independent_integrated_KMS',r['kms_actual']-r['kms_expected'],3e-11)
 ck('true_absorption_vs_relaxation',r['relaxation']['value']/(r['j']['value']+r['chi']['value'])-1,2e-14)
 for n in('j','chi'):
  assert r[n]['representable']and r[n]['value']>0
  ck('ordinary_log_'+n,math.log(r[n]['value'])-r[n]['log'],3e-12)
 # Independent closed-form collinear support crossings: E=(X²+ml²)/(2X)
 # with X=Ua-Ub or X=2Enu+Ua-Ub, provided X>0.
 D=r['state']['potential_a']-r['state']['potential_b'];ml=r['lepton_mass']
 for X in(D,2*r['Enu']+D):
  if X>0:
   edge=(X*X+ml*ml)/(2*X)
   if ml<edge<r['upper_El']:ck('exact_support_endpoint_energy',min(abs(edge-x)for x in r['panels_El']),2e-9)
for n in('j','chi'):
 ck('electron8_to12_q_'+n,e8[n]['value']/ef[n]['value']-1,1e-6)
 ck('mu12_to16_cos_'+n,mu[-2][n]['value']/mu[-1][n]['value']-1,1e-6)
 ck('mu_cos16_q16_'+n,mu[-1][n]['value']/q16[n]['value']-1,1e-6)
 ck('mu_q20_q24_'+n,q20[n]['value']/q24[n]['value']-1,1e-6)
 ck('mu_cos20_q24_'+n,cos20[n]['value']/q24[n]['value']-1,1e-6)
 ck('mu_inner_order_'+n,next(r for r in controls if r['probe']=='mu_inner32')[n]['value']/mu[0][n]['value']-1,1e-7)
 ck('mu_tails_'+n,next(r for r in controls if r['probe']=='mu_tail80')[n]['value']/mu[0][n]['value']-1,1e-6)
 ck('zero_field_common939_free_'+n,anchor[0][n]['value']/anchor[1][n]['value']-1,1e-6)
scope='Selected nu_e and nu_mu capture/inverse ONLY; GM1 external forward OPEN, no full antinu inventory/RPA/production claim'
old=json.loads((P/'exploratory_unpartitioned_results.json').read_text());coarse=[]
for r in old:
 if r['flavor']=='mu':coarse.append(dict(order=r['controls']['lepton_order'],relative_error=r['chi']['value']/q20['chi']['value']-1,qualified=False,reason='Exploratory unpartitioned grid; refinement failed'))
retained=[dict(name='mu_q16_q20',error=abs(q16['chi']['value']/q20['chi']['value']-1),limit=1e-6,passed=False)]
for name in('zero_field_initial8_results.json','zero_field_initial12_results.json'):
 a=json.loads((P/name).read_text());retained.append(dict(name=name,error=abs(a[0]['chi']['value']/a[1]['chi']['value']-1),limit=1e-6,passed=False))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),retained_exploratory_muon_failures=coarse,retained_refinement_failures=retained,scope=scope)
(P/'validation_results.json').write_text(json.dumps(out,indent=2))
sources=[ROOT/x for x in['gm1_differential_response/adapter.py','rmf_reference/gm1_finite_t.py','rmf_phase/stability.py','relativistic_tensor/tensor.py','muon_full_current/current.py','muon_vacuum/vacuum_cc.py','thermodynamics/lepton_reference.py']]
files=list(P.glob('*'))+sources
manifest=dict(scope=scope,status='AUTHOR_FROZEN_PENDING_INDEPENDENT_REVIEW',environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()for f in files if f.is_file()and f.name!='CHECKPOINT.json'})
(P/'CHECKPOINT.json').write_text(json.dumps(manifest,indent=2));print(out['passed'],out['failed'])
if out['failed']:raise SystemExit(1)
