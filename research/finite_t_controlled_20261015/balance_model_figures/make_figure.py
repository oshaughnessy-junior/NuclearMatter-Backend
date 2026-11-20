"""Plot frozen selected-bath evidence; imports no physical or driver modules."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; O=Path(__file__).resolve().parent
paths=['allowed_rate_assessment/results.json','transparent_balance_grid/summary.json','transparent_capture_balance/summary.json','review_allowed_rate_assessment/REVIEW.md','review_transparent_balance_grid/REVIEW.md']
pins={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths}
a=json.loads((R/paths[0]).read_text()); grid=json.loads((R/paths[1]).read_text()); anchor=json.loads((R/paths[2]).read_text())
rows=[]
for x in a['rows']:
 g=anchor if x['label']=='original_anchor' else next(z for z in grid['states'] if z['T']==x['T'] and z['nB']==x['nB'])
 assert abs(g['kinetic_Ye']-x['detailed_kinetic_Ye'])<1e-12
 errors={c:list(x['channels'][c]['relative_error']) for c in ('EC','PC','beta')}
 for c in errors:
  assert np.allclose(errors[c],np.asarray(x['channels'][c]['allowed'])/np.asarray(x['channels'][c]['detailed'])-1,rtol=0,atol=3e-15)
 rows.append(dict(T=x['T'],nB=x['nB'],kinetic=g['kinetic_Ye'],chemical=g['chemical_Ye'],allowed=x['allowed_kinetic_Ye'],errors=errors))
rows.sort(key=lambda x:(x['nB'],x['T']))
labels=[f"{x['T']:g} MeV\n{x['nB']:.0e} fm$^{{-3}}$" for x in rows];xx=np.arange(len(rows))
fig,axs=plt.subplots(1,2,figsize=(11,4.5))
for key,lab,marker,color in [('kinetic','Detailed kinetic balance','o','#176f8c'),('allowed','Allowed-model kinetic balance','s','#b36917'),('chemical','Chemical-potential equality','^','#654c9c')]:
 axs[0].scatter(xx,[x[key] for x in rows],label=lab,marker=marker,color=color,s=45,zorder=3)
axs[0].set(ylabel='Electron fraction $Y_e$',title='Balance depends on the physical rate model',ylim=(.23,.64));axs[0].legend(fontsize=8)
for c,color in [('EC','#176f8c'),('PC','#b36917'),('beta','#654c9c')]:
 for j,mark,lab in [(0,'o','number'),(1,'s','energy')]:
  axs[1].scatter(xx+(j-.5)*.1,[100*x['errors'][c][j] for x in rows],marker=mark,color=color,s=32,label=f'{c} {lab}',zorder=3)
axs[1].axhspan(-1,1,color='gray',alpha=.12,label='Selected 1% assessment gate');axs[1].axhline(0,color='gray',lw=.7)
axs[1].set(ylabel='Allowed / detailed − 1 [%]',title='Moment errors at each detailed kinetic root');axs[1].legend(fontsize=6,ncol=2,loc='upper left')
for ax in axs:
 ax.set_xticks(xx,labels);ax.grid(axis='y',alpha=.2);ax.set_xlabel('Fixed temperature and baryon number density')
fig.suptitle('Free pair-npe matter; beta + electron / positron capture; empty incoming neutrinos',fontsize=11)
fig.text(.5,.015,'Selected thermostatted baths, not thermal equilibrium. Numerical convergence is distinct from model error.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.94))
for ext in ('png','pdf'):fig.savefig(O/f'balance_model_comparison.{ext}',dpi=180)
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in pins.items())
(O/'figure_data.json').write_text(json.dumps(dict(source_sha256=pins,rows=rows,scope='Saved accepted selected-bath evidence only; no new physical quadrature, fit or evolution.',arithmetic_checks=16),indent=2)+'\n')
print('16 saved-data identity checks pass; figure and data saved')
