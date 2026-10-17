from pathlib import Path
import json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
new=json.loads((P/'results.json').read_text());old=json.loads((P.parent/'urca_threshold/initial_cold_extension_failure.json').read_text());oldrow=next(x for x in old['rows']if x['T']==.01 and x['reference']['Yp']==.1);row=new['rows'][0];ref=row['methods'][1]['log_chi']
labels=['Frozen mapped48 (excluded)','Frozen mapped72 (excluded)','Resolved Ee/cosine16','Resolved Ee/cosine24','Adaptive y/q32']
logs=[oldrow['fine_logs'][1],oldrow['log_chi'],row['methods'][2]['log_chi'],row['methods'][3]['log_chi'],row['methods'][0]['log_chi']]
errors=[abs(math.expm1(v-ref))for v in logs]
plt.rcParams.update({'font.size':10})
fig,ax=plt.subplots(figsize=(8.4,3.8));ax.barh(range(5),errors,color=['#aa4433']*2+['#327ab0']*3);ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlim(1e-16,.03);ax.axvline(3e-6,color='black',ls='--',lw=1,label='Unchanged acceptance gate');ax.set_xlabel('Relative difference in chi from adaptive y/q64');ax.set_title('Selected cold capture: resolving quadrature, preserving the old failure');ax.legend(loc='lower right',frameon=False);fig.text(.98,.015,'nB=0.01 fm^-3, Yp=0.1, T=0.01 MeV, Eν=0.03 MeV; same free-NR physical response',ha='right',fontsize=8);fig.tight_layout(rect=(0,.045,1,1));fig.savefig(P/'cold_convergence.png',dpi=170);fig.savefig(P/'cold_convergence.pdf');(P/'figure_data.json').write_text(json.dumps(dict(labels=labels,relative_errors=errors,reference_log_chi=ref,gate=3e-6),indent=2)+'\n')
