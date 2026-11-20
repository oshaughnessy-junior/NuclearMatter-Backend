"""Plot frozen selected heavy-family results; no physical-module imports."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=Path(__file__).resolve().parent
names=['heavy_baryon_limit_assessment/comparison.csv','review_heavy_baryon_limit_assessment/REVIEW.md','heavy_baryon_limit_assessment/LOCK.json'];pins={p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in names}
rows=[{k:float(v) for k,v in x.items()} for x in csv.DictReader((R/names[0]).open())];assert [x['lambda'] for x in rows]==[1,2,4,8]
cols=['R_EC_fm-3s-1','Q_EC_MeVfm-3s-1','R_PC_fm-3s-1','Q_PC_MeVfm-3s-1'];err=['R_EC_relative_allowed','Q_EC_relative_allowed','R_PC_relative_allowed','Q_PC_relative_allowed'];anchor={k:rows[0][k]/(1+rows[0][e]) for k,e in zip(cols,err)}
q=rows[0]['mn_MeV']-rows[0]['mp_MeV'];mbar=(rows[0]['mn_MeV']+rows[0]['mp_MeV'])/2
for x in rows:
 assert abs(x['mn_MeV']-x['mp_MeV']-q)<2e-12
 assert abs((x['mn_MeV']+x['mp_MeV'])/2/mbar-x['lambda'])<3e-15
 for k,e in zip(cols,err):assert abs(x[k]/anchor[k]-1-x[e])<3e-15
fig,axes=plt.subplots(1,2,figsize=(10,4.7),sharey=True)
for ax,c,title in zip(axes,('EC','PC'),('Electron capture','Positron capture')):
 for prefix,lab,marker,color in [('R','Number','o','#176f8c'),('Q','Neutrino energy','s','#b76a1b')]:
  ax.plot([1/x['lambda'] for x in rows],[100*x[f'{prefix}_{c}_relative_allowed'] for x in rows],marker=marker,color=color,label=lab)
 ax.axhline(0,color='gray',lw=.9,ls='--');ax.set(title=title,xlim=(0,1.06),ylim=(-6.2,.35));ax.set_xticks([.125,.25,.5,1],['1/8','1/4','1/2','1']);ax.set_xlabel('Inverse artificial common mass scale 1/λ');ax.grid(alpha=.2)
axes[0].set_ylabel('Point-current phase / stationary allowed − 1 [%]');axes[0].legend(loc='lower left',fontsize=9)
fig.suptitle('Fixed-Q heavy-baryon family: T = 3 MeV, nB = 10⁻⁷ fm⁻³, Ye = 0.54901',fontsize=11)
fig.text(.5,.025,'Four tested scales; lines guide the eye. No fitted asymptotic law or physical mass-uncertainty claim.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.94))
for ext in ('png','pdf'):fig.savefig(O/f'heavy_baryon_limit.{ext}',dpi=180)
assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in pins.items())
(O/'figure_data.json').write_text(json.dumps(dict(source_sha256=pins,T_MeV=3,nB_fm3=1e-7,Ye=.5490065797957709,Q_MeV=q,point_current='F1=1, GA=1.2723, F2=GP=0',allowed_anchors=anchor,rows=rows,saved_arithmetic_checks=24,scope='Four reviewed artificial-mass points; no new physical integration/fit/uncertainty/AD.'),indent=2)+'\n')
print('24 saved-data identities PASS; source hashes unchanged')
