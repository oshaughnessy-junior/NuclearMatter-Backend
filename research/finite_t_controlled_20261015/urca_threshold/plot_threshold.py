"""Plot only the nine qualified spectra; no phenomenological threshold fit."""
from pathlib import Path
import json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'qualified_results.json').read_text())
assert r['status']=='PASS'
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
colors={.03:'#277DA1',.1:'#F8961E',.3:'#43AA8B'}
for T,color in colors.items():
    rows=sorted([x for x in r['rows'] if x['T']==T],key=lambda x:x['reference']['delta_p'])
    axes[0].semilogy([x['reference']['delta_p'] for x in rows],[math.exp(x['log_chi_over_T2']) for x in rows],'o--',color=color,label=f'T={T:g} MeV')
axes[0].axvline(0,color='.5',lw=1)
axes[0].set(xlabel=r'Cold momentum deficit $p_{Fn}-p_{Fp}-p_{Fe}$ [MeV]',ylabel=r'$\chi(E_\nu=3T)/T^2$ [s$^{-1}$ MeV$^{-2}$]',title='Selected local models around the cold triangle')
axes[0].legend(frameon=False)
for Y,style,label in [(.1,'o-','Yp=0.10 (below triangle)'),(.12,'s-','Yp=0.12 (above triangle)')]:
    rows=sorted([x for x in r['rows'] if x['reference']['Yp']==Y],key=lambda x:x['T'])
    axes[1].loglog([x['T'] for x in rows],[math.exp(x['log_chi_over_T2']) for x in rows],style,label=label)
critical=json.loads((HERE/'cold_support_results.json').read_text())['critical_Enu_MeV']/3
axes[1].axvline(critical,color='.5',ls=':',label=r'Yp=.10 cold $E_\nu$ onset / 3')
axes[1].set(xlabel='T [MeV], with Eν=3T',ylabel=r'$\chi/T^2$ [s$^{-1}$ MeV$^{-2}$]',title='Thermal opening plus finite-neutrino kinematics')
axes[1].legend(frameon=False,fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Constant-offset NR quasiparticle ansatz; lines guide selected points',fontsize=11)
fig.savefig(HERE/'threshold_selected.png',dpi=180)
fig.savefig(HERE/'threshold_selected.pdf')
