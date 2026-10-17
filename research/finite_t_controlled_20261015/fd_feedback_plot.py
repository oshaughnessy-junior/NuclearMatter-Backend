"""Plot accepted short FD element snapshots; no continuous-solution claim."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/nmb-finite-t-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
source=ROOT/'fd_fluid_element/short_evolution_results.json'
x=json.loads(source.read_text())
OUT=ROOT/'papers';OUT.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for name,color,marker in (('coarse','#bb4a30','s'),('fine','#18789b','o')):
    run=x[name];initial=run['rows'][0]['before']
    ts=[0.];states=[initial];t=0.
    for row in run['rows']:
        t+=row['dt'];ts.append(t*1e6);states.append(row['after'])
    for ax,key,scale in zip(axes,('T','u','Ye'),(1e3,1e3,1.)):
        ax.plot(ts,[(s[key]-initial[key])*scale for s in states],color=color,marker=marker,
                ms=5,lw=1.3,ls='--' if name=='coarse' else '-',mfc='none' if name=='coarse' else color,
                label=f"{run['steps']} Euler steps")
for ax,label,title in zip(axes,('Temperature change [keV]','Matter energy change [keV/baryon]','Electron fraction change'),('Composition relaxation heats','Total matter energy falls','Weak composition feedback')):
    ax.set(xlabel='Time [microseconds]',ylabel=label,title=title)
    ax.grid(alpha=.18)
axes[0].legend(frameon=False)
fig.suptitle('Transparent fixed-density n,p,e± element; no photons')
for ext in ('png','pdf'):fig.savefig(OUT/f'fd_feedback_snapshots.{ext}',dpi=180)
payload={'scope':'Two/four explicit Euler snapshots, not a converged long-time solution','nB_fm3':x['nB'],'photons':x['photons'],'duration_s':x['duration_s'],'coarse':x['coarse'],'fine':x['fine'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
(OUT/'fd_feedback_snapshots.json').write_text(json.dumps(payload,indent=2)+'\n')
print('Figure and exact snapshot data saved')
