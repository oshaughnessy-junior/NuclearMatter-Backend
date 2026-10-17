"""Plot preserved selected dilute-capture convergence, without rerunning physics."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/nmb-finite-t-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];P=Path(__file__).resolve().parent
paths=[R/'relativistic_capture_integrals'/n for n in ['integration_results.json','heldout_controls_results.json','anchor_results.json']]
a,b,c=[json.loads(p.read_text()) for p in paths];ref=c[-1]['chi_s'];rows=[x for x in a if x['label']=='dilute'];rows.append([x for x in b if x['state']['temperature']==.2][-1]);data=dict(reference_chi_s=ref,gate=3e-6,source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},rows=[dict(order=x['controls']['lepton_order'],controls=x['controls'],chi_s=x['chi']['value'],relative_error=abs(x['chi']['value']/ref-1)) for x in rows],scope='SelectedT.2,nB1e-15,Ye.3,Ymu0,E_nu150MeV; joint controls; numerical comparison not model uncertainty')
(P/'measurements.json').write_text(json.dumps(data,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(8,4.3));xx=[x['order'] for x in data['rows']];yy=[x['relative_error'] for x in data['rows']];ax.plot(xx,yy,color='#555555',lw=1)
for x,y in zip(xx,yy):ax.scatter(x,y,color='#a94332' if y>data['gate'] else '#287fb0',s=65,zorder=3)
ax.axhline(data['gate'],color='black',ls='--',label='Selected pilot gate: 3e-6');ax.set_yscale('log');ax.set_xticks(xx);ax.set_xlabel('Joint outer quadrature order (other controls also refined)');ax.set_ylabel('Relative difference from dilute FD vacuum-flux anchor');ax.set_title('Muon capture: preserved coarse failures and resolved reference');ax.grid(alpha=.2);ax.legend(frameon=False);fig.tight_layout();fig.savefig(P/'capture_convergence.png',dpi=170);fig.savefig(P/'capture_convergence.pdf');print(json.dumps({'points':len(xx),'first_error':yy[0],'final_error':yy[-1]}))
