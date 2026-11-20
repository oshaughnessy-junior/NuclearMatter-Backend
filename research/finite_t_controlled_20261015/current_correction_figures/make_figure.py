"""Plot immutable reviewed current-slice measurements; no physics imports."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
inputs=['current_correction_assessment/comparison.csv','review_current_correction_assessment/REVIEW.md']
pins={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in inputs}
rows=[]
for a in csv.DictReader((R/inputs[0]).open()):
 x={k:(v if k in ('channel','moment') else float(v)) for k,v in a.items()}
 for k in ('baseline','F2zero','allowed'):assert x[k]>0
 assert abs(x['F2zero']/x['baseline']-1-x['F2_removal_fraction'])<2e-15
 assert abs(x['allowed']/x['baseline']-1-x['allowed_over_baseline_minus1'])<2e-15
 assert abs(x['allowed']/x['F2zero']-1-x['allowed_over_F2zero_minus1'])<2e-15
 assert abs((1+x['allowed_over_F2zero_minus1'])*(1+x['F2_removal_fraction'])-(1+x['allowed_over_baseline_minus1']))<2e-15
 rows.append(x)
fig,axes=plt.subplots(1,2,figsize=(9.5,4.5),sharey=True)
for ax,c,title in zip(axes,('EC','PC'),('Electron capture','Positron capture')):
 rr=[x for x in rows if x['channel']==c];assert [x['moment'] for x in rr]==['number','neutrino_energy']
 xx=np.arange(2)
 ax.bar(xx-.17,[x['F2zero']/x['baseline']-1 for x in rr],bottom=1,width=.32,color='#1c7893',label='Only Pauli F2 term removed')
 ax.bar(xx+.17,[x['allowed']/x['baseline']-1 for x in rr],bottom=1,width=.32,color='#b56a20',label='Stationary allowed model')
 ax.axhline(1,color='#333333',lw=1,label='Detailed full-current reference')
 ax.set_xticks(xx,['Number emission','Neutrino energy emission']);ax.set_title(title);ax.set_ylim(.9,1.16);ax.grid(axis='y',alpha=.17)
 for i,x in enumerate(rr):
  for dx,k in [(-.17,'F2zero'),(.17,'allowed')]:
   y=x[k]/x['baseline'];ax.text(i+dx,y+(.004 if y>=1 else -.017),f'{100*(y-1):+.2f}%',ha='center',fontsize=9)
axes[0].set_ylabel('Moment / detailed full-current moment')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.89),ncol=1,fontsize=9)
fig.suptitle('One controlled free-matter bath: T = 3 MeV, nB = 10⁻⁷ fm⁻³, Ye = 0.54901',fontsize=11,y=.99)
fig.text(.5,.025,'F1, GA and GP remain fixed in the F2-only comparison. This is not a physical uncertainty band.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.75))
for ext in ('png','pdf'):fig.savefig(O/f'current_slice_comparison.{ext}',dpi=180)
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in pins.items())
(O/'figure_data.json').write_text(json.dumps(dict(T_MeV=3,nB_fm3=1e-7,Ye=.5490065797957709,source_sha256=pins,rows=rows,saved_arithmetic_checks=16,scope='Selected reviewed saved moments; no new physical quadrature/evolution or model uncertainty.'),indent=2)+'\n')
print('16 saved-moment arithmetic identities PASS; frozen inputs unchanged')
