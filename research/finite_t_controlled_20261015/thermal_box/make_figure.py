from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/nmb-thermal-box-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;r=json.loads((P/'results.json').read_text());eight=json.loads((P/'equilibrium_grid.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharex=True)
for i,ax in enumerate(axes):
 for f,edges,label,color,ls in [(r['initial']['f'][i],r['grid']['edges'],'Initial radiation','#777777',':'),(r['runs'][-1]['final']['f'][i],r['grid']['edges'],'Eight-step endpoint','#287cb1','-'),(r['LTE']['f'][i],r['grid']['edges'],'4-bin constrained LTE','#b04b36','--'),(eight['LTE']['f'][i],eight['grid']['edges'],'8-bin constrained LTE','#46965e','-.')]:ax.stairs(f,edges,label=label,color=color,linestyle=ls,lw=1.8)
 ax.set_yscale('log');ax.set_ylim(1e-9,1.1);ax.set_xlabel('Neutrino energy [MeV]');ax.set_title(['Electron neutrino','Electron antineutrino'][i]);ax.grid(alpha=.2)
axes[0].set_ylabel('Bin occupation');axes[1].legend(fontsize=8,frameon=False,loc='lower left');fig.suptitle('Closed finite-grid box: short exchange and constrained stationary states',fontsize=11);fig.tight_layout();fig.savefig(P/'finite_box.png',dpi=170);fig.savefig(P/'finite_box.pdf')
