"""Compare saved spectral channels and their empty-radiation moment ledgers.

This script imports no physical kernel or evidence driver. It does arithmetic
on separately computed spectra; it does not evolve a joined radiation field.
"""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
HC = 197.3269804
LOG_DOS = math.log(2 * math.pi**2 * HC**3)
PATHS = {
    "EC": [f"hot_ec_opacity_spectrum/total_fine_part{i}.json" for i in range(2)],
    "PC": [f"hot_pc_opacity_spectrum/total_fine_{i}.json" for i in range(4)],
    "beta": [f"hot_beta_opacity_spectrum/total_fine_{i}.json" for i in range(2)],
}
records, moments, hashes = {}, {}, {}
for channel, names in PATHS.items():
    rows = []
    for name in names:
        raw = (BASE / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        rows.extend(json.loads(raw)["rows"])
    rows.sort(key=lambda r: r["E"])
    assert len({r["index"] for r in rows}) == len(rows)
    logs, energy_logs, values = [], [], []
    for row in rows:
        E, weight, source = row["E"], row["weight"], row["spectrum"]
        assert E > 0 and weight > 0
        state = source["matched_state"]
        assert state["T"] == 2.0 and state["nB"] == 1e-7 and state["Ye"] == .3
        for direction in ("j", "chi"):
            value = source[direction]["log"]
            assert not isinstance(value, bool) and value is not None and math.isfinite(value)
        log_density = 2 * math.log(E) - LOG_DOS + source["j"]["log"]
        logs.append(math.log(weight) + log_density)
        energy_logs.append(math.log(weight) + math.log(E) + log_density)
        values.append({"E_MeV": E, "log_number_emission_density": log_density,
                       "log_j": source["j"]["log"], "log_chi": source["chi"]["log"]})
    records[channel] = values
    moments[channel] = {"R": math.exp(float(logsumexp(logs))),
                        "Q": math.exp(float(logsumexp(energy_logs)))}

name = "three_source_cooling_element/results.json"
raw = (BASE / name).read_bytes()
hashes[name] = hashlib.sha256(raw).hexdigest()
reference = json.loads(raw)["rows"][-1]["records"][0]["source"]
checks = []
for channel, result in moments.items():
    for field, source_field in (("R", "number"), ("Q", "neutrino_energy")):
        truth = reference["channels"][channel][source_field]
        error = result[field] / truth - 1
        checks.append({"channel": channel, "field": field, "relative_error": error,
                       "gate": 3e-6, "passed": abs(error) < 3e-6})
net = moments["beta"]["R"] + moments["PC"]["R"] - moments["EC"]["R"]
Q = math.fsum(row["Q"] for row in moments.values())
for field, value in (("R", net), ("Q", Q)):
    error = value / reference[field] - 1
    checks.append({"channel": "named_three_channel_sum", "field": field,
                   "relative_error": error, "gate": 3e-6, "passed": abs(error) < 3e-6})
assert all(c["passed"] for c in checks)

fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), layout="constrained")
labels = {"EC": "EC: electron neutrinos", "PC": "PC: electron antineutrinos",
          "beta": "Neutron beta: electron antineutrinos"}
for ax in axes:
    for channel, rows in records.items():
        ax.plot([r["E_MeV"] for r in rows],
                [r["log_number_emission_density"] / math.log(10) for r in rows],
                marker=".", ms=3, label=labels[channel])
    ax.axvline(.7820077025790402, ls=":", color="0.4", lw=1)
    ax.set_xlabel("Neutrino / antineutrino energy [MeV]")
    ax.set_ylabel("Number emission per energy [fm⁻³ s⁻¹ MeV⁻¹]")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: rf"$10^{{{int(x)}}}$"))
    ax.set_ylim(-20, -6)
    ax.grid(alpha=.15)
axes[0].set_xlim(0, 1.25)
axes[0].set_title("Soft spectrum; dotted line is neutron-rest endpoint")
axes[1].set_xlim(0, 35)
axes[1].set_title("Same channels over the capture spectrum")
axes[1].legend(fontsize=8)
fig.suptitle("Actual saved spectra: T=2 MeV, nB=10⁻⁷ fm⁻³, Ye=0.3; empty incoming radiation")
fig.savefig(HERE / "three_channel_spectra.png", dpi=180)
fig.savefig(HERE / "three_channel_spectra.pdf")
data = {"scope": "Saved selected-state spectra and empty-radiation arithmetic only; no joined evolution/full inventory",
        "source_sha256": hashes, "spectra": records, "moments": moments,
        "net_composition_source": net, "escaping_energy_source": Q,
        "checks": checks, "plot_note": "Connecting lines guide the eye; tails outside plotted range remain nonzero."}
(HERE / "figure_data.json").write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps({"moment_checks_passed": len(checks), "net_R": net, "Q": Q,
                  "max_separate_channel_error": max(abs(c["relative_error"]) for c in checks[:6])}))
