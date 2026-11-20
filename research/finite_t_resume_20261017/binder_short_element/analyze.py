"""Additional source-record accounting, no EOS/source calls or trajectory changes."""
from pathlib import Path
import json,math
P=Path(__file__).resolve().parent;load=lambda f:json.loads((P/f).read_text());rows=[]
for steps in [2,4]:
 p=load('rung_'+str(steps)+'.json');thermal=composition=0.;n=1e-7
 for entry in p['history']:
  r=load('group_'+str(entry['group'])+'.json');s=r['state'];d=r['rhs'];h=entry['h']
  thermal+=h*s['cv_fixed_nB_Ye']*d['temperature_dot_mev_s'];composition+=h*s['denergy_dYe_fixed_nB_T']*d['Ye_dot_s']
 b=p['endpoint'];rows.append(dict(steps=steps,integrated_Cv_Tdot=thermal,integrated_eY_Ydot=composition,sum_plus_escaped=thermal+composition+p['escaped_energy'],endpoint_baryon_density=b['neutron']['particle_density']+b['proton']['particle_density'],endpoint_baryon_relative=(b['neutron']['particle_density']+b['proton']['particle_density'])/n-1,endpoint_Le_material=n*p['Ye'],escaped_Le=p['escaped_nue']-p['escaped_anti'],initial_Le_material=n*.3,Le_absolute_residual=n*p['Ye']+p['escaped_nue']-p['escaped_anti']-n*.3))
(P/'ADDITIONAL_ACCOUNTING.json').write_text(json.dumps(dict(rows=rows,qualification='Diagnostic decompositions of saved actual stage integrals, no extra prelocked pass gate or independent evolution reference'),indent=2)+'\n')
