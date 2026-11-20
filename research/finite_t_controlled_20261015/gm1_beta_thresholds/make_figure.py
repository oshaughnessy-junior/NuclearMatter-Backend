from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;d=json.loads((p/'results.json').read_text());n=[s['nB']for s in d['rows']];fig,axes=plt.subplots(2,1,figsize=(7.2,6.4),sharex=True,constrained_layout=True)
for key,label,color in [('Yp','protons','#222222'),('Ye','electrons','#1565c0'),('Ymu','muons','#b23a48')]:axes[0].plot(n,[s[key]for s in d['rows']],label=label,color=color,lw=2)
for key,label,color in [('delta_e_MeV','electron triangle','#1565c0'),('delta_mu_MeV','muon triangle','#b23a48')]:axes[1].plot(n,[s[key]for s in d['rows']],label=label,color=color,lw=2)
for ax in axes:
 for name,color in [('e','#1565c0'),('mu','#b23a48')]:ax.axvline(d['onsets'][name]['state']['nB'],color=color,ls='--',lw=1,alpha=.7)
 ax.grid(alpha=.2);ax.legend(frameon=False,loc='best')
axes[1].axhline(0,color='#555555',lw=1);axes[0].set_ylabel('Particle / net lepton fraction');axes[1].set_ylabel(r'$p_{Fn}-p_{Fp}-p_{F\ell}$ [MeV]');axes[1].set_xlabel(r'$n_B$ [fm$^{-3}$]');axes[0].set_title('Cold GM1 beta equilibrium: common bare mass 939 MeV\nDashed lines: zero-neutrino-momentum Urca triangles; no rates')
fig.savefig(p/'cold_beta_thresholds.png',dpi=170);fig.savefig(p/'cold_beta_thresholds.pdf')
