from reference import *
checks=[];rows=[];start=time.monotonic()
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
def snap():
 files=[p for p in (ROOT/'hot_beta_opacity_spectrum').iterdir()if p.is_file()]+[ROOT/x for x in S.PINS]+[ROOT/'review_hot_beta/REVIEW.md',ROOT/'review_numerics/crossed_review.md']
 return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
before=snap();(P/'before.json').write_text(json.dumps(before,indent=2)+'\n')
ck('commissioned spectrum SHA',int(before['hot_beta_opacity_spectrum/spectrum.py']!='e5b67fd4206bf3f766225e7030864729213b29944deda17fbdb606a9630957ff'),0)
ck('commissioned manifest SHA',int(before['hot_beta_opacity_spectrum/MANIFEST.json']!='6f7cc97e9d5820da6ff7d0592691b5f438a7292daf98eb6fe295aea7660c9af8'),0)
ck('hot total record immutable pin',int(before['hot_beta_moments/results.json']!='57e2e89e820b0e80b9d0cd4956d67c54e208cd78d5e2609edcd8e1ef54f94d88'),0)
# New physical rest events first, independent trace versus literal reciprocal spinors.
for ee,c in [(.63,-.71),(.92,.23),(1.17,.62)]:
 n,p,l,k,*_=S.B.electron_events(np.array([ee]),np.array([c]),MN,MP,ME)
 tr=trace_amp(n,p,l,k)[0];ck('absolute trace vs decay spinors '+str(ee),tr/S.B.amplitude(n,p,l,k)[0]-1,3e-12);ck('absolute trace vs inverse spinors '+str(ee),tr/S.B.amplitude(n,p,l,k,inverse=True)[0]-1,3e-12)
for en in [.31,.835]:
 vals=[]
 for controls in [(8,10,10,8,65.),(12,16,16,12,85.)]:
  v=integral(en,*controls);vals.append(v);rows.append(dict(kind='independent',value=v));print('independent',en,controls,v['seconds'],flush=True)
 ref=S.integrate(en,parent_order=14,cone_order=18,electron_order=18,phi_order=12,tail=85.);rows.append(dict(kind='author_fresh',value=ref))
 for key in ['j','chi']:
  ck('independent rule refinement '+str(en)+' '+key,math.expm1(vals[0][key+'_log']-vals[1][key+'_log']),3e-6)
  ck('independent fresh phase '+str(en)+' '+key,math.expm1(vals[1][key+'_log']-ref[key]['log']),3e-6)
 ck('new positive thermal Doppler support '+str(en),int(not all(math.isfinite(vals[1][key+'_log'])for key in ['j','chi'])),0)
 ck('independent shell/energy '+str(en),vals[1]['max_energy_residual'],3e-10)
 mu=S.FIXED['mu_n']-S.FIXED['mu_p']-S.FIXED['mu_e'];ck('KMS after separate occupations '+str(en),vals[1]['j_log']-vals[1]['chi_log']-(mu-en)/2,3e-11)
ck('all frozen source bytes unchanged',int(before!=snap()),0)
(P/'pilot_results.json').write_text(json.dumps(dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,seconds=time.monotonic()-start),indent=2)+'\n');print('DONE',len(checks),[c for c in checks if not c['passed']],time.monotonic()-start)
