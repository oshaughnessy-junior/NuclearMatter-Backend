"""Plot frozen soft-PC support and saved coefficients; no rate integration."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'pc_support' / 'SEMI_INFINITE_RADIAL_RAW.json'
raw_bytes = SOURCE.read_bytes()
data = json.loads(raw_bytes)
rows = [r for r in data['rows'] if r['rule']['radial'] == 32]
assert len(rows) == 2
assert all(r['ordinary_j_s-1'] is None and r['ordinary_true_chi_s-1'] is None for r in rows)
mn, mp, me = 939.56542052, 938.27208816, .51099895069
threshold = ((mn + me)**2 - mp**2) / (2 * mp)
energies = np.linspace(.05, threshold, 500)
D = ((mn + me)**2 - mp**2) / (2 * energies)
emin = (D**2 + mp**2) / (2 * D)
for r in rows:
    d = ((mn + me)**2 - mp**2) / (2 * r['E_MeV'])
    anchor = (d*d + mp*mp) / (2*d)
    assert abs(anchor - r['threshold_initial_proton_energy_MeV']) < 1e-8

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.2))
ax = axs[0]
ax.plot(energies, emin / 1000, color='#215a8a', label='Exact inverse-PC support threshold')
ax.scatter([r['E_MeV'] for r in rows], [r['threshold_initial_proton_energy_MeV']/1000 for r in rows], color='#b34331', zorder=3, label='Selected computed energies')
ax.axhline(mp/1000, color='.45', linestyle=':', label='Proton rest energy')
ax.axvline(threshold, color='.45', linestyle='--')
ax.set(xlabel=r'Antineutrino energy $E_\nu$ [MeV]', ylabel='Minimum initial proton energy [GeV]', yscale='log', xlim=(0, 1.92))
ax.set_title('Moving targets retain soft positive support')
ax.legend(fontsize=8, loc='upper right')
ax = axs[1]
ax.scatter([r['E_MeV'] for r in rows], [r['log_true_chi_s-1'] for r in rows], s=48, marker='o', color='#215a8a', label=r'True absorption $\chi$')
ax.scatter([r['E_MeV'] for r in rows], [r['log_j_s-1'] for r in rows], s=65, marker='x', color='#b34331', label=r'Spontaneous emission $j$')
for r in rows:
    dy = 12 if r['log_true_chi_s-1'] < -3000 else -18
    ax.annotate(f"{r['log_true_chi_s-1']:.3f}", (r['E_MeV'], r['log_true_chi_s-1']), xytext=(5, dy), textcoords='offset points', fontsize=8)
ax.set(xlabel=r'Antineutrino energy $E_\nu$ [MeV]', ylabel=r'$\ln[\mathrm{coefficient}/(1\,\mathrm{s}^{-1})]$', xlim=(.085, .175))
ax.set_title('Finite logs; ordinary values underflow')
ax.legend(fontsize=8, loc='lower right')
fig.text(.5, .015, 'Author points: 0.10/0.15 MeV. Separate integrated checks: 0.10/0.13 MeV. Fixed T = 2 MeV; no all-energy or transport claim.', ha='center', fontsize=8)
fig.tight_layout(rect=(0, .06, 1, 1))
fig.savefig(HERE/'soft_pc_support.png', dpi=180)
fig.savefig(HERE/'soft_pc_support.pdf')
(HERE/'SOURCE_DATA.json').write_text(json.dumps({'status': 'Selected frozen author values; different integrated checks at0.10/0.13, not a new0.15 independent integral', 'source': str(SOURCE), 'source_sha256': hashlib.sha256(raw_bytes).hexdigest(), 'candidate_sha256': data['candidate_sha256'], 'different_review_sha256':'fe591a912ec7210b4f8897a3714b67c95299134d68efb76eb27409d917d0a7f6', 'rows': rows, 'masses_MeV': {'mn':mn,'mp':mp,'me':me}, 'static_target_threshold_MeV': threshold, 'new_rate_integrations':0, 'interpolation_or_fit_of_rates':False}, indent=2)+'\n')
print('Saved selected support/coefficients figure from frozen data; no physical integration.')
