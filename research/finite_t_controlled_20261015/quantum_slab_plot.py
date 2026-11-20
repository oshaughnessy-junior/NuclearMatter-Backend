"""Plot evaluated optical depths only; no spectral interpolation."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'quantum_slab/validation_results.json').read_text())['selected_energy_tables']
fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
colors={'dilute':'#18789b','moderate_fd':'#bb4a30'}
tags=sorted(set(r['state']['tag'] for r in rows))
colors=dict(zip(tags,('#18789b','#bb4a30')))
for ax,flavor,title in zip(axes,('nue','antinue'),(r'$\nu_e$',r'$\bar\nu_e$')):
    for tag in tags:
        selected=[r for r in rows if r['state']['tag']==tag and r['coefficients']['flavor']==flavor]
        for key,marker,fill in (('tau_relaxation','o',True),('tau_true_absorption','s',False)):
            ax.scatter([r['coefficients']['energy_MeV'] for r in selected],
                       [r['fixed_1km_slab'][key] for r in selected],s=46,marker=marker,
                       facecolors=colors[tag] if fill else 'none',edgecolors=colors[tag],linewidths=1.3)
    ax.axhline(1,color='.65',lw=.8,ls='--')
    ax.set(xscale='log',yscale='log',xlabel='Neutrino energy [MeV]',title=title)
    ax.grid(alpha=.15)
axes[0].set_ylabel('Optical depth across 1 km')
handles=[Line2D([],[],marker='o',ls='',color=colors[t],label=t.replace('_',' ')) for t in tags]
handles += [Line2D([],[],marker='o',ls='',color='.25',label='Relaxation (j+chi)/c'),
            Line2D([],[],marker='s',ls='',color='.25',mfc='none',label='True absorption chi/c')]
fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
fig.suptitle('Matched free-FD bath at T=2 MeV: selected energies')
(ROOT/'papers').mkdir(exist_ok=True)
for ext in ('png','pdf'):
    fig.savefig(ROOT/'papers'/f'quantum_slab_depths.{ext}',dpi=180)
print(tags)
