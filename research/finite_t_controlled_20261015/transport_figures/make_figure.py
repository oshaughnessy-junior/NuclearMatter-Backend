"""Plot preserved Lorentz-gas transport measurements, without rate refitting."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

b=Path(__file__).resolve().parent.parent
p=b/'screened_coulomb_transport/results.json'
d=json.loads(p.read_text())
selected=[r['result'] for r in d['rows'] if r['kind']=='selected_state']
ladder=next(r['values'] for r in d['rows'] if r['kind']=='degenerate_ladder')
ratios=[r['kappa_open']/r['kappa_Estar_zero'] for r in selected]
assert all(0<x<=1 for x in ratios)
fig,ax=plt.subplots(1,2,figsize=(10.3,4.6),layout='constrained')
labels=[f"$n_e={r['state']['ne_fm3']:.0e}$\n$T={r['state']['T']:g}, Z={r['state']['Z']}$" for r in selected]
ax[0].bar(range(4),ratios,color=['#d17c42','#4f94a8','#4f94a8','#4f94a8'],width=.65)
for i,r in enumerate(ratios):ax[0].text(i,r+.025,f'{r:.3f}',ha='center',fontsize=10)
ax[0].set(xticks=range(4),xticklabels=labels,ylim=(0,1.15),ylabel=r'$\kappa_{j=0}/(K_2/T)$',title='The electrical constraint changes heat flow')
ax[0].tick_params(axis='x',labelsize=8)
x=np.array([r['T_over_TF'] for r in ladder]);y=np.array([r['WF_relative'] for r in ladder])
ax[1].loglog(x,y,'o-',color='#315f83',label='Calculated moments')
ax[1].loglog(x,y[-1]*(x/x[-1])**2,'--',color='#bf7646',label=r'$T^2$ guide, anchored at lowest $T$')
ax[1].set(xlabel=r'$T/T_F$',ylabel=r'$\kappa/(\sigma T L_0)-1$',title='Approach to Wiedemann–Franz')
ax[1].legend(fontsize=8,loc='upper left')
ax[1].text(.05,.11,r'$n_e=10^{-7}$ fm$^{-3}$, $Z=1$',transform=ax[1].transAxes,fontsize=10)
for a in ax:a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
fig.suptitle('Static screened-ion Lorentz gas: selected physical limits',fontsize=13)
fig.supxlabel(r'$n_e$ in fm$^{-3}$; $T$ in MeV.  $L_0=\pi^2/(3e_{\rm HL}^2)$.  Uncorrelated external ions; local screening.',fontsize=9)
out=b/'transport_figures'
for ext in ['png','pdf']:fig.savefig(out/f'lorentz_transport_limits.{ext}',dpi=180)
rows={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selected':selected,'degenerate_ladder':ladder,'open_circuit_fraction':ratios,'scope':'Static uncorrelated external ions; Born/local susceptibility screening; selected points only; lines join measured points, T² guide is not fitted conductivity.'}
(out/'lorentz_transport_measurements.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'open_circuit_fraction':ratios,'WF_relative':list(y)}))
