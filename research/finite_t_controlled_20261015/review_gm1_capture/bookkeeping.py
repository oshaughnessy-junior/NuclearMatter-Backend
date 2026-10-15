from pathlib import Path
import sys,json,math,hashlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'gm1_capture_integrals'));import capture as C
checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for name in ['mu_q16.json','mu_q20.json','mu_q24.json','mu_cos20.json']:
 d=json.loads((ROOT/'gm1_capture_integrals'/name).read_text());j,chi,relax=(d[k]['value']for k in ['j','chi','relaxation']);gate(name+' true chi separated from relaxation',relax/(j+chi)-1,2e-14);gate(name+' independently reported KMS',math.log(j/chi)-d['kms_expected'],3e-12);rows.append(dict(file=name,j=j,chi=chi,relaxation=relax))
q16,q20,q24,cos20=[json.loads((ROOT/'gm1_capture_integrals'/n).read_text())for n in ['mu_q16.json','mu_q20.json','mu_q24.json','mu_cos20.json']]
coarse=q16['chi']['value']/q20['chi']['value']-1;fine=q20['chi']['value']/q24['chi']['value']-1;coordinate=cos20['chi']['value']/q24['chi']['value']-1
gate('earlier muon q refinement still fails original1e-6',float(abs(coarse)<=1e-6),0.);gate('later muon q refinement unchanged1e-6',fine,1e-6);gate('muon coordinate comparison unchanged1e-6',coordinate,1e-6)
s=C.A.matter_state(8,.22,.17,.04).ordered();r=C.A.differential(s,40.,180.,-.3,C.A.leptons.ELECTRON_MASS,203.11989906077872,energy_order=24,phi_order=4,tail=80);mp=r['response']['mapping'];q=mp['q'];w=mp['kinetic_omega'];omega=mp['physical_omega']
gate('physical energy transfer',omega-(40-180),1e-13);gate('kinetic energy transfer includes potential difference',w-(omega-s.potential_b+s.potential_a),1e-13);gate('form factor invariant physical',mp['physical_q2']-(omega*omega-q*q),1e-10);gate('vertex invariant kinetic',mp['kinetic_qstar2']-(w*w-q*q),1e-10);gate('bare mass separate',s.weak_bare_mass-939,0.)
# The free zero-field anchors explicitly retain common939 and no physical vacuum Q.
z=json.loads((ROOT/'gm1_capture_integrals/zero_field_results.json').read_text());
for i,r0 in enumerate(z):
 st=r0['state'];gate(f'zero field{i} common939',st['dirac_mass']-939,0.);gate(f'zero field{i} vectors',abs(st['potential_a'])+abs(st['potential_b']),0.)
rows.append(dict(zero_field_evidence=z,mapping_probe=mp,muon_coarse_relative=coarse,muon_fine_relative=fine,muon_coordinate_relative=coordinate))
(HERE/'bookkeeping.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks))
