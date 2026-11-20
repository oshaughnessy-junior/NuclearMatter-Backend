"""Plot frozen coefficient predictions and measured reference errors only."""
from pathlib import Path
import json,sys,importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('figure_locked_fit_api',P/'api.py');A=importlib.util.module_from_spec(sp);sys.modules[sp.name]=A;sp.loader.exec_module(A);r=json.loads((P/'validation_results.json').read_text());E=np.geomspace(1,30,160);fit=[A.evaluate(float(e))for e in E]
fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for key,color,label in [('rate','#286BA4','Total'),('transport_rate','#D47B28','Transport')]:
 axes[0].loglog(E,[z[key]for z in fit],color=color,label=label+' fit')
 h=[z for z in r['rows']if z['part']=='heldout'];axes[0].scatter([z['E']for z in h],[z['reference'][key]for z in h],color=color,s=24,marker='o')
axes[0].set(xlabel='Incident energy [MeV]',ylabel='Rate per incident νe [s⁻¹]',title='Positive moments; dots are heldouts');axes[0].legend(frameon=False,fontsize=8)
for key,color,label in [('rate','#286BA4','Total'),('transport_rate','#D47B28','Transport'),('outgoing_energy_moment','#50845C','Outgoing energy')]:
 h=[z for z in r['rows']if z['part']=='heldout'];axes[1].semilogx([z['E']for z in h],[100*z['errors'][key]for z in h],'o-',color=color,label=label)
axes[1].axhspan(-1,1,color='gray',alpha=.1);axes[1].axhline(0,color='gray',lw=.7);axes[1].set(xlabel='Heldout energy [MeV]',ylabel='Measured fit error [%]',ylim=(-1.1,1.1),title='Fixed 1% positive-moment gate');axes[1].legend(frameon=False,fontsize=7)
for part,marker,label in [('training','o','Training'),('heldout','s','Heldout')]:
 rr=[z for z in r['rows']if z['part']==part];axes[2].semilogx([z['E']for z in rr],[z['errors']['signed_RT']for z in rr],marker,ls='none',label=label)
axes[2].axhspan(-.03,.03,color='gray',alpha=.15);axes[2].axhline(0,color='gray',lw=.7);axes[2].set(xlabel='Incident energy [MeV]',ylabel=r'$(\Delta Q_{fit}-\Delta Q_{ref})/(R_{ref}T)$',title='Signed-energy gate: two retained failures');axes[2].legend(frameon=False,fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Fixed pair bath T=2 MeV, μe=0.174463332 MeV; νe on e− AND e+; outgoing ν unblocked\nLocked training-only Chebyshev fits, 1–30 MeV; measured points do not certify uniform error',fontsize=10)
fig.savefig(P/'moment_fit_validation.png',dpi=160);fig.savefig(P/'moment_fit_validation.pdf')
