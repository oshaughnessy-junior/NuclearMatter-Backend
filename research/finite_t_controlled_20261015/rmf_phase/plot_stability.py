"""Plot sampled nodes only; no interpolation or binodal implied."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nmb-rmf-phase-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

root=Path(__file__).resolve().parent
r=json.loads((root/'validation_results.json').read_text());data=r['map']
values=[s['eigenvalues_mev_fm3'][0] for s in data]
norm=TwoSlopeNorm(vmin=min(values),vcenter=0.,vmax=max(values))
fig,axes=plt.subplots(1,3,figsize=(11.8,4.2),sharex=True,sharey=True)
for ax,T in zip(axes,[1.,10.,20.]):
 rows=[s for s in data if s['T']==T]
 artist=ax.scatter([s['nB'] for s in rows],[s['Yp'] for s in rows],
                   c=[s['eigenvalues_mev_fm3'][0] for s in rows],cmap='RdBu',norm=norm,s=95,marker='s')
 for s in rows:
  if s['fixed_Y_mechanically_stable'] and s['local_label']=='spinodal':
   ax.scatter(s['nB'],s['Yp'],s=165,facecolors='none',edgecolors='black',linewidths=1.5)
 ax.set_title(f'T = {T:g} MeV');ax.set_xlabel(r'$n_B$ [fm$^{-3}$]');ax.set_xlim(.005,.337)
 ax.set_xticks([.02,.08,.16,.24,.32]);ax.set_yticks([.1,.3,.5]);ax.grid(alpha=.15)
axes[0].set_ylabel(r'$Y_p$')
fig.subplots_adjust(left=.065,right=.88,bottom=.26,top=.82,wspace=.07)
cax=fig.add_axes([.9,.26,.015,.56])
fig.colorbar(artist,cax=cax,label=r'Lowest chemical-Hessian eigenvalue [MeV fm$^3$]')
fig.suptitle('Uniform GM1: local nuclear-only stability at sampled nodes',fontsize=13)
fig.text(.065,.045,'Red: spinodal. Blue: locally stable. Black ring: fixed-composition compressibility is positive despite a negative eigenvalue.\nNo lepton/Coulomb contribution, global phase construction or interpolation between nodes.',fontsize=8.5)
fig.savefig(root/'local_stability_nodes.png',dpi=180,bbox_inches='tight')
fig.savefig(root/'local_stability_nodes.pdf',bbox_inches='tight')
print(root/'local_stability_nodes.png')
