"""Plot immutable finite slab output; nodes/face flux, no interpolated physics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).parent;r=json.loads((P/'accepted_results.json').read_text());g=r['grids'][-1];a=r['accepted_run'];E=np.array(g['E']);mu=np.array(g['mu']);W=np.array(g['W']);f=np.array(a['final']);f0=np.array(a['initial']);en=E[::2];nc=len(f)
fig,ax=plt.subplots(1,2,figsize=(9.5,3.7),layout='constrained')
ax[0].plot(en,en**2*f0[0].reshape(-1,2).mean(axis=1),'k--o',label='Initial radiation')
for i,row in enumerate(f):ax[0].plot(en,en**2*row.reshape(-1,2).mean(axis=1),'o-',label=f'Cell {i+1}')
ax[0].set(xlabel='Neutrino energy [MeV]',ylabel=r'$E^2\langle f\rangle_\mu$ [MeV$^2$]',title='After 0.06 s; four actual energy nodes');ax[0].legend(frameon=False,fontsize=8)
from scipy.special import expit
left=expit((a['rad_mu']-E)/a['left_T']);right=expit((a['rad_mu']-E)/a['right_T']);faces=np.empty((nc+1,len(E)));faces[0]=np.where(mu>0,left,f[0]);faces[-1]=np.where(mu>0,f[-1],right)
for i in range(1,nc):faces[i]=np.where(mu>0,f[i-1],f[i])
flux=faces@(W*mu*E);ax[1].plot(np.arange(nc+1)/nc,flux,'o-');ax[1].set(xlabel=r'Slab coordinate $z/L$',ylabel=r'$F_{\nu,E}/c$ [MeV fm$^{-3}$]',title='Final signed upwind face energy flux')
for x in ax:x.grid(alpha=.25)
fig.suptitle('Finite 0–24 MeV S₂ slab: three cells, pair bath T=2 MeV\nIncoming FD radiation: left T=3, right T=1.3 MeV; μν=−0.7 MeV',fontsize=10)
fig.savefig(P/'slab_feedback.png',dpi=160);fig.savefig(P/'slab_feedback.pdf')
(P/'figure_data.json').write_text(json.dumps(dict(energy_nodes=en.tolist(),final_cell_E2meanf=[(en**2*row.reshape(-1,2).mean(axis=1)).tolist()for row in f],face_z_over_L=(np.arange(nc+1)/nc).tolist(),final_flux_over_c=flux.tolist(),qualification='Finite S2 threecell fourenergy-node data. Lines guide the eye; no continuum profile or convergence claim.'),indent=2))
