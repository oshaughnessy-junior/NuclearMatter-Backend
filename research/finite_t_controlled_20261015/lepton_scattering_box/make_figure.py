"""Plot only frozen detailed-grid outputs; no new physics or fit."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent;r=json.loads((P/'refined_edge_grid_results.json').read_text());E=np.array(r['grid']['E'])
fig,axs=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
for ax,run in zip(axs,r['runs']):
 ax.plot(E,E**2*np.array(run['initial']),'o--',color='#2E6F9E',label='Initial radiation')
 ax.plot(E,E**2*np.array(run['final']),'s-',color='#CC5B36',label='After 0.08 s')
 ax.set(xlabel='Neutrino energy E [MeV]',ylabel=r'$E^2 f_\nu(E)$ [MeV$^2$]',title=f"Initial radiation T={run['Tr']:g} MeV")
 ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
fig.suptitle('Isotropic νe–e± scattering: fixed T=2 MeV pair bath\n8 physical energy nodes; closed 0–24 MeV box',fontsize=11)
fig.savefig(P/'spectral_feedback.png',dpi=160);fig.savefig(P/'spectral_feedback.pdf')
