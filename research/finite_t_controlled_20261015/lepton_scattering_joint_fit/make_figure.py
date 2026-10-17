"""Measured validation errors and declared cancellation guard, no uniform certificate."""
from pathlib import Path
import json,importlib.util,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;s=importlib.util.spec_from_file_location('joint_plot_api',P/'api.py');A=importlib.util.module_from_spec(s);sys.modules[s.name]=A;s.loader.exec_module(A)
v=json.loads((P/'validation.json').read_text());tr=json.loads((P/'training.json').read_text());E=np.geomspace(1,30,501);zz=[A.evaluate(float(e),method='diagnostic_fit')for e in E];fig,ax=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
ax[0].plot(E,[z['g']for z in zz],label='Locked joint fit');ax[0].scatter([r['E']for r in tr['rows']],[r['fine']['total']['neutrino_energy_change']/(2*r['fine']['total']['rate'])for r in tr['rows']],s=13,label='Known development');ax[0].scatter([r['E']for r in v['rows']],[r['fine']['total']['neutrino_energy_change']/(2*r['fine']['total']['rate'])for r in v['rows']],s=23,marker='x',label='Fresh validation');ax[0].set_xscale('log');ax[0].axhline(0,color='k',lw=.5);ax[0].set(xlabel='Incident energy [MeV]',ylabel=r'$g=\Delta Q/(RT)$',title='Signed physical energy transfer');ax[0].legend(fontsize=8)
for key,label in [('rate','R'),('transport_rate','Transport R'),('outgoing_energy_moment','Qout')]:ax[1].plot([r['E']for r in v['rows']],[100*(r['raw_locked_fit'][key]/r['fine']['total'][key]-1)for r in v['rows']],marker='o',ms=3,label=label)
for a in[-1,1]:ax[1].axhline(a,color='gray',ls='--',lw=.8)
ax[1].set(xlabel='Incident energy [MeV]',ylabel='Fit relative error [%]',title='Positive moments: fresh points');ax[1].legend(fontsize=8)
for r in v['rows']:
 er=(r['raw_locked_fit']['neutrino_energy_change']-r['fine']['total']['neutrino_energy_change'])/(2*r['fine']['total']['rate']);ax[2].scatter(r['E'],er,color='C3'if r['auto']['fallback']else'C0',marker='s'if r['auto']['fallback']else'o')
for a in[-.005,.005]:ax[2].axhline(a,color='gray',ls='--',lw=.8)
ax[2].set(xlabel='Incident energy [MeV]',ylabel=r'$(\Delta Q_{fit}-\Delta Q_{ref})/(R_{ref}T)$',title='Raw signed fit; red = actual fallback')
fig.suptitle('Fixed T=2 MeV pair bath; νe, unblocked outgoing radiation — measured points only',fontsize=11);fig.savefig(P/'joint_moments.png',dpi=180);fig.savefig(P/'joint_moments.pdf')
