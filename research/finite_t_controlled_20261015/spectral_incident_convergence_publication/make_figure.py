"""Plot saved incident-rule convergence; performs no weak-rate calculations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--review-sha', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = root/'spectral_incident_convergence/CANONICAL_FINITE_RANGE_RESULTS.json'
review = root/'review_spectral_incident_convergence/REVIEW.md'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '4ecd574c54174b461726891971a1b0b7fa597d134a6d27297f3fcc8fb940a22c'
assert hashlib.sha256(review.read_bytes()).hexdigest() == args.review_sha
data = json.loads(source.read_text())
rows = data['profile_aggregates']
assert all(x['whole_moment_qualified'] is False for x in rows)
orders = [4, 8, 16, 24]
channels = ['EC', 'PC', 'NEUTRON_DECAY']
labels = ['Electron capture: 2–6 MeV', 'Positron capture: 2–6 MeV', 'Neutron decay: 0.2–0.8 MeV']
curves = {}
for channel in channels:
    selected = [r for r in rows if r['channel'] == channel]
    ref = {r['radiation']['name']: r for r in selected if r['order'] == 32}
    assert len(ref) == 3
    curves[channel] = []
    for order in orders:
        errors = [[], []]
        for row in selected:
            if row['order'] != order:
                continue
            baseline = ref[row['radiation']['name']]
            for branch in ['forward', 'backward']:
                for k in range(2):
                    a, b = row[branch][k], baseline[branch][k]
                    if a['log'] is None or b['log'] is None:
                        assert row['radiation']['mode'] == 'empty' and branch == 'backward'
                        assert a['value'] == b['value'] == 0
                        continue
                    errors[k].append(abs(math.expm1(a['log']-b['log'])))
        curves[channel].append({'order': order, 'number': max(errors[0]), 'energy': max(errors[1])})

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.1), sharey=True)
for ax, channel, title in zip(axes, channels, labels):
    for key, color, marker, label in [('number', '#25476a', 'o', 'Number'), ('energy', '#b85b2d', 's', 'Energy')]:
        y = [max(row[key], np.finfo(float).eps) for row in curves[channel]]
        ax.semilogy(orders, y, color=color, marker=marker, linewidth=1.6, markersize=5, label=label)
    ax.axhline(3e-6, color='#666666', linestyle='--', linewidth=1, label='Adjacent-rule target')
    ax.set_title(title, fontsize=10.5)
    ax.set_xticks(orders)
    ax.set_xlabel('Incident Gauss nodes')
    ax.grid(axis='y', alpha=.18)
axes[0].set_ylabel('Worst positive F/B moment difference\nrelative to the 32-node rule')
axes[0].set_ylim(1e-16, 2e-2)
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .89),
           ncol=3, frameon=False, fontsize=9)
fig.suptitle('Finite-band incident quadrature at fixed inner controls', fontsize=13, y=.98)
fig.text(.5, .035, 'One prescribed T = 2 MeV bath; maximum over three radiation closures. Omitted energy ranges and tails remain open.', ha='center', fontsize=9)
fig.text(.5, .006, 'Plot floor is machine epsilon only; no physical coefficient floor or whole-energy qualification.', ha='center', fontsize=8, color='#555555')
fig.tight_layout(rect=(0, .09, 1, .92))
out = Path(__file__).resolve().parent
fig.savefig(out/'finite-t-spectral-incident-convergence-2026-10-16.png', dpi=180)
fig.savefig(out/'finite-t-spectral-incident-convergence-2026-10-16.pdf')
(out/'figure_data.json').write_text(json.dumps({'kind': 'SAVED incident-rule differences only', 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'review_sha256': args.review_sha, 'fixed_inner_controls': data['controls'], 'bath': data['bath'], 'curves': curves, 'whole_energy_moments_tails': 'OPEN', 'new_physical_calls': 0}, indent=2, allow_nan=False)+'\n')
print(json.dumps({'output': str(out), 'curves': curves}))
