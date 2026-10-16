from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;fig,axes=plt.subplots(1,3,figsize=(10.5,3.2),layout='constrained')
for n in [2,4,8]:
 z=json.loads((P/f'trajectory_{n}.json').read_text());t=[e['t']*1e3 for e in z['endpoints']];axes[0].plot(t,[e['state']['T']for e in z['endpoints']],'-o',ms=3,label=str(n)+'steps');axes[1].plot(t,[e['state']['Ye']for e in z['endpoints']],'-o',ms=3);axes[2].plot(t,[e['source']['Q']/z['endpoints'][0]['source']['Q']for e in z['endpoints']],'-o',ms=3)
for a in axes:a.set_xlabel('Prescribed time [ms]');a.grid(alpha=.2)
axes[0].set_ylabel('Temperature [MeV]');axes[1].set_ylabel(r'$Y_e$');axes[2].set_ylabel(r'Actual $Q_\nu(t)/Q_\nu(0)$');axes[0].legend(fontsize=8);fig.suptitle('Short free-npe weak-on element: prescribed compression, trapped ideal LTE photons',fontsize=10);fig.savefig(P/'weak_density.png',dpi=160);fig.savefig(P/'weak_density.pdf');plt.close(fig)
