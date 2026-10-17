"""Render frozen calculated cell temperatures, no new physical approximation."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).parent;d=json.loads((H/'results.json').read_text());fig,axes=plt.subplots(1,2,figsize=(9.6,4.4),layout='constrained')
for ax,label,title in zip(axes,['isolated','reservoir'],['Insulated faces: heat redistributes','Fixed reservoirs: boundary heat flow']):
 r=next(x['result']for x in d['rows']if x.get('label')==label and x['result']['controls']['steps']==64);N=len(r['theta']);x=(np.arange(N)+.5)/N
 ax.plot(x,r['initial_theta'],'o--',color='#999999',label='Initial')
 ax.plot(x,r['theta'],'o-',color='#316487',label=r'Calculated $\tau=0.02$')
 if label=='reservoir':
  s=next(x['steady']for x in d['rows']if'steady'in x and x['steady']['N']==8);ax.plot(np.r_[0,x,1],np.r_[1.08,s['theta'],.92],':',color='#b6673f',label='Independent steady integral')
 ax.set(xlabel=r'$x/L$',ylabel=r'$T/T_0$',title=title,ylim=(.87,1.13));ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Electron heat diffusion in a fixed screened-ion background',fontsize=13)
fig.supxlabel(r'$T_0=0.041$ MeV; $n_e=5.3\times10^{-8}$ fm$^{-3}$; $Z=2$.  Eight cells; local FD imposed; electron $C_V$ only.',fontsize=9)
for ext in['png','pdf']:fig.savefig(H/f'thermal_profiles.{ext}',dpi=180)
(H/'figure_provenance.json').write_text(json.dumps(dict(source_results_sha256=hashlib.sha256((H/'results.json').read_bytes()).hexdigest(),scope='Actual 8-cell short-time profiles; joined cell values, not a continuum accuracy claim. Steady curve is independent integral of actual conductivity.'),indent=2)+'\n')
