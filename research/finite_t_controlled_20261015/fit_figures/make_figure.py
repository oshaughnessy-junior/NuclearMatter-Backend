"""Compare frozen potential fits on the new fit's saved blocked EOS states."""
from pathlib import Path
import importlib.util
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
b=Path(__file__).resolve().parent.parent
p=b/'gm1_affinity_fit/holdout_results.json'
d=json.loads(p.read_text())
spec=importlib.util.spec_from_file_location('frozen_old_potential',b/'gm1_potential_fit/potential.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
rows=[]
for r in d['rows']:
 e=r['exact'];assert e['max_normalized_equation_residual']<1e-8
 q=old.evaluate(r['n'],r['T'],r['Y'])
 ref=e['mu_n_mev']-e['mu_p_mev'];err=(q['mu_n']-q['mu_p'])-ref
 rows.append(dict(group=r['group'],n=r['n'],T=r['T'],Y=r['Y'],old_balance=abs(math.expm1(err/r['T'])),new_balance=r['errors']['balance'],old_delta_mu=err,new_delta_mu=r['errors']['delta_mu']))
fig,ax=plt.subplots(figsize=(7.4,4.8),layout='constrained')
for label,key,c,m in [('Original 60 coefficients','old_balance','#999999','o'),('New 196 coefficients','new_balance','#316487','s')]:
 ax.scatter([r['T'] for r in rows],[100*r[key] for r in rows],marker=m,color=c,s=34,alpha=.8,label=label)
ax.axhline(1,color='#b96e41',linestyle='--',label='Declared 1% balance-factor gate')
ax.set(xscale='log',yscale='log',xlabel='Temperature [MeV]',ylabel=r'$100\,|\exp(\delta\mu_{np}/T)-1|$ [%]',title='One-potential affinity accuracy at 30 new blocked states')
ax.set_xticks([.12,.2,.5,1,2,3],labels=['0.12','0.2','0.5','1','2','3'])
ax.legend(fontsize=9,loc='lower left');ax.grid(alpha=.18)
fig.supxlabel(r'Common-939 nuclear GM1; $0.18\leq n_B\leq0.26$ fm$^{-3}$, $0.18\leq Y_p\leq0.33$.  Sampled states, not a uniform bound.',fontsize=8)
out=b/'fit_figures'
for ext in ['png','pdf']:fig.savefig(out/f'affinity_fit_holdouts.{ext}',dpi=180)
source_paths=['gm1_affinity_fit/holdout_results.json','gm1_affinity_fit/coefficients.json','gm1_affinity_fit/potential.py','gm1_potential_fit/coefficients.json','gm1_potential_fit/potential.py']
(out/'affinity_fit_measurements.json').write_text(json.dumps(dict(source_sha256={x:hashlib.sha256((b/x).read_bytes()).hexdigest() for x in source_paths},rows=rows,scope='Both frozen fits evaluated on new-fit blocked samples; old failed development point excluded. New basis and coefficients frozen before these actual EOS states. Errors compare same lepton contribution/temperature, not actual weak emissivity.'),indent=2)+'\n')
print(json.dumps(dict(max_old_percent=100*max(r['old_balance'] for r in rows),max_new_percent=100*max(r['new_balance'] for r in rows),rows=len(rows))))
