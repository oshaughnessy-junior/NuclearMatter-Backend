"""Render frozen changing-state trajectories and measured temporal differences."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=Path(__file__).resolve().parent.parent
ap=b/'transparent_antinu_element/results.json';rp=b/'review_transparent_element/results.json'
a=json.loads(ap.read_text());r=json.loads(rp.read_text());assert not a['failed'] and not r['failed']
runs=a['runs'];heun=r['heun'];e=runs[-1];h=heun[-1]
et=[x['time'] for x in e['records']]+[e['duration']];ey=[x['state']['Ye'] for x in e['records']]+[e['final']['Ye']]
ht=np.linspace(0,e['duration'],len(h['records'])+1);hy=[x['Ye'] for x in h['records']]+[h['final']['Ye']]
fig,ax=plt.subplots(1,2,figsize=(10.0,4.5),layout='constrained')
ax[0].plot(np.array(et)*1000,ey,'o--',color='#aa8058',label='Euler, 4 steps')
ax[0].plot(ht*1000,hy,'s-',color='#316487',markersize=4,label='Independent Heun, 8 steps')
ax[0].set(xlabel='Time [ms]',ylabel=r'Net electron fraction $Y_e$',title='Actual sources respond to changing matter')
ax[0].legend(fontsize=8);ax[0].text(.05,.73,f"Heun final $T$ = {h['final']['T']:.7f} MeV",transform=ax[0].transAxes,fontsize=9)
diffs={}
for label,rr,color,marker in [('Euler',runs,'#aa8058','o'),('Heun',heun,'#316487','s')]:
 x=np.array([q['steps'] for q in rr[1:]]);y=np.abs(np.diff([q['final']['Ye'] for q in rr]));diffs[label]={'finer_steps':x.tolist(),'successive_Ye_change':y.tolist()}
 ax[1].loglog(x,y,marker+'-',color=color,label=label)
 slope=1 if label=='Euler' else 2
 ax[1].loglog(x,y[-1]*(x/x[-1])**(-slope),':',color=color,label=f'$N^{{-{slope}}}$ guide')
ax[1].set(xlabel='Finer step count $N$',ylabel=r'$|Y_e(N)-Y_e(N/2)|$',title='Conservation accuracy is not time accuracy')
ax[1].set_xticks([2,4,8],labels=['2','4','8']);ax[1].legend(fontsize=8)
for x in ax:x.grid(alpha=.18)
fig.suptitle('Restricted transparent fluid element: beta decay + positron capture',fontsize=12)
fig.supxlabel(r'$n_B=10^{-7}$ fm$^{-3}$; initially $T=2$ MeV, $Y_e=0.3$.  Free npe pair EOS; no electron capture, neutrino-pair source or radiation evolution.',fontsize=8)
o=b/'element_figures'
for ext in ['png','pdf']:fig.savefig(o/f'transparent_element_feedback.{ext}',dpi=180)
(o/'element_measurements.json').write_text(json.dumps({'source_sha256':{str(p.relative_to(b)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ap,rp]},'Euler4':{'time_s':et,'Ye':ey,'final_T_MeV':e['final']['T']},'Heun8':{'time_s':ht.tolist(),'Ye':hy,'final_T_MeV':h['final']['T']},'successive_grid_changes':diffs,'scope':'Named beta+PC channels only; actual changing sources, prescribed empty radiation, not complete beta balance. Guides anchored to measured last difference, no fitted trajectory or exact-solution claim.'},indent=2)+'\n')
print(json.dumps(diffs))
