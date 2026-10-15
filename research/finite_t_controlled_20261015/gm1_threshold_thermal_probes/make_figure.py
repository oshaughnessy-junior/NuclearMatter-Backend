"""Selected actual coefficient points; joining lines are not fitted widths."""
from pathlib import Path
import json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;r=json.loads((P/'results.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
for factor,color in zip((.97,1.,1.03),('#c66a16','#2365a3','#27805b')):
 frozen=sorted([x for x in r['rows']if x['density_factor']==factor and x['closure'].startswith('frozen')],key=lambda x:x['T'])
 warm=sorted([x for x in r['rows']if x['density_factor']==factor and x['closure'].startswith('selfconsistent')],key=lambda x:x['T'])
 T=[x['T']for x in frozen];y=[x['runs'][-1]['chi']['value']for x in frozen];z=[x['runs'][-1]['chi']['value']for x in warm]
 label=f'{factor:.2f} ncrit; cold window '+('empty'if factor<1 else'open')
 axes[0].plot(T,y,'o-',color=color,label=label);axes[0].plot(T,z,'x--',color=color)
 axes[1].plot(T,[100*(b/a-1)for a,b in zip(y,z)],'o-',color=color,label=f'{factor:.2f} ncrit')
for ax in axes:ax.set_xscale('log');ax.set_xlabel('Temperature [MeV]');ax.set_xticks([.1,.3,1.],['0.1','0.3','1']);ax.grid(alpha=.2)
axes[0].set_yscale('log');axes[0].set_ylabel('Differential absorption chi\n[s$^{-1}$ MeV$^{-1}$ per unit cosine]');axes[0].legend(fontsize=8,loc='lower right')
axes[0].set_title('Actual finite-T response, fixed cold fractions')
axes[1].set_ylabel('100 (self-consistent warm / frozen fields - 1) [%]');axes[1].axhline(0,color='gray',lw=.8);axes[1].legend(fontsize=9)
axes[1].set_title('Field-closure effect at the same densities')
fig.suptitle('GM1 electron-neutrino absorption probes: Enu=0.3 MeV, El=cold mu_e, cos(theta)=-1\n'+f'Cold DU triangle ncrit={r["cold_onset"]["nB"]:.6f}; selected finite-Enu cold opening={r["finite_incident_selected_probe_cold_onset"]:.6f} fm$^{{-3}}$',fontsize=11)
fig.supxlabel('Solid circles: frozen cold fields with thermal density roots; dashed crosses: warm self-consistent fields.\nLines join three computed points; no fitted width, integrated Urca rate, or external GM1 validation.',fontsize=8)
fig.savefig(P/'selected_thermal_probes.png',dpi=180);fig.savefig(P/'selected_thermal_probes.pdf')
