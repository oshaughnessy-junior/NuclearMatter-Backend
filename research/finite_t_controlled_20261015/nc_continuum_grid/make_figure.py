from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/nmb-thermal-box-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;base=json.loads((P/'results.json').read_text());fine=json.loads((P/'dense_grid_refinement.json').read_text());fig,axes=plt.subplots(1,3,figsize=(11,3.8),sharey=True)
for ax,row,rr in zip(axes,base['rows'],fine['rows']):
 for kind,color in [('uniform','#a54d32'),('resolved','#287cb0')]:
  vals=[x for x in row['grids']if x['kind']==kind];ax.plot([len(x['nodes'])for x in vals],[max(x['relative_errors'])for x in vals],'o-',color=color,label=kind+' q96')
 for q,color,marker in [(192,'#42844e','s'),(384,'#8555a0','*')]:
  vals=[x for x in rr['grids']if x['angle_order']==q];ax.plot([len(x['nodes'])for x in vals],[max(x['relative_errors'])for x in vals],marker+'-',color=color,label='resolved q'+str(q))
 ax.axhline(3e-5,color='black',ls='--',lw=1);ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(1e-10,20);ax.set_title(row['label']);ax.set_xlabel('Outgoing energy nodes');ax.grid(alpha=.2)
axes[0].set_ylabel('Largest relative error across7moments');axes[2].legend(fontsize=8,frameon=False,loc='lower left');fig.suptitle('NC continuum redistribution: energy and angular resolution',fontsize=11);fig.tight_layout();fig.savefig(P/'continuum_grid_errors.png',dpi=170);fig.savefig(P/'continuum_grid_errors.pdf')
