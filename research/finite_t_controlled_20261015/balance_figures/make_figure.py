"""Plot frozen refined source measurements; imports no physical driver."""
from pathlib import Path
import json, hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
p=B/'transparent_capture_balance/results.json';raw=p.read_bytes();d=json.loads(raw)
assert d['passed']==328 and not d['failed']
rows=[r['refined'] for r in d['rows']]+[d['balanced'],d['chemical_source']]
points={float(r['Ye']):dict(Ye=float(r['Ye']),dYe_dt=float(r['dYe_dt']),Q=float(r['escaping_energy']))for r in rows}
a=sorted(points.values(),key=lambda x:x['Ye'])
yk=float(d['kinetic_balance_Ye']);yc=float(d['chemical_balance_Ye'])
fig,ax=plt.subplots(1,2,figsize=(12.4,5.6),layout='constrained')
for z,key,label in zip(ax,['dYe_dt','Q'],[r'$dY_e/dt$ [s$^{-1}$]',r'Escaping energy [MeV fm$^{-3}$ s$^{-1}$]']):
 z.plot([r['Ye']for r in a],[r[key]for r in a],'o-',color='#304c67',ms=5,label='Actual refined source calls')
 z.axvline(yk,color='#137c9e',ls='--',label=rf'Kinetic balance $Y_e={yk:.6f}$')
 z.axvline(yc,color='#bd6827',ls=':',label=rf'Chemical balance $Y_e={yc:.6f}$')
 z.set_xlabel(r'Prescribed composition $Y_e$');z.set_ylabel(label);z.grid(alpha=.2)
ax[0].axhline(0,color='.3',lw=.8);ax[0].legend(fontsize=9,loc='upper right')
ax[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax[0].set_title('Composition source and distinct roots')
ax[1].set_title('Positive radiation loss at kinetic balance')
fig.suptitle(r'Transparent free-npe/e$^\pm$ source balance: $T=2$ MeV, $n_B=10^{-7}$ fm$^{-3}$'+'\nEmpty incoming neutrinos; fixed-temperature thermostat and rapid EM pair equilibration',fontsize=13)
fig.supxlabel('Refined evaluated points; connecting lines guide the eye. No cooling evolution or full neutrino inventory.',fontsize=9)
for ext in ['png','pdf']:fig.savefig(B/f'balance_figures/transparent_balance.{ext}',dpi=170)
e=dict(source='transparent_capture_balance/results.json',source_sha256=hashlib.sha256(raw).hexdigest(),kinetic_Ye=yk,chemical_Ye=yc,measurements=a)
(B/'balance_figures/figure_data.json').write_text(json.dumps(e,indent=2)+'\n')
print(json.dumps(dict(points=len(a),kinetic_Ye=yk,chemical_Ye=yc)))
