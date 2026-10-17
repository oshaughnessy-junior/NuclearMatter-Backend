from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;a=json.loads((P/'cycle_32.json').read_text());b=json.loads((P/'cycle_64.json').read_text());fig,ax=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
for z,label,ls in [(a,'RK4 32','--'),(b,'RK4 64','-')]:
 t=[r['time']for r in z['records']];ax[0].plot(t,[r['nB']/1e-7 for r in z['records']],ls,label=label);ax[1].plot(t,[r['T']for r in z['records']],ls,label=label);s0=z['records'][0]['entropy_per_baryon'];ax[2].plot(t,[(r['entropy_per_baryon']/s0-1)*1e9 for r in z['records']],ls,label=label)
roots=b['entropy_roots'];ax[1].plot([r['time']for r in roots],[r['T_root']for r in roots],'o',ms=4,label='Entropy roots')
for a in ax:a.set_xlabel('Prescribed time [s]');a.grid(alpha=.2)
ax[0].set_ylabel(r'$n_B/n_0$');ax[1].set_ylabel('Temperature [MeV]');ax[2].set_ylabel(r'$(s/n_B)/(s_0/n_0)-1$ [$10^{-9}$]');ax[1].legend(fontsize=8);fig.suptitle('Weak-off free npe + trapped LTE photons: prescribed compression–expansion cycle',fontsize=10)
fig.savefig(P/'density_cycle.png',dpi=160);fig.savefig(P/'density_cycle.pdf');plt.close(fig)
