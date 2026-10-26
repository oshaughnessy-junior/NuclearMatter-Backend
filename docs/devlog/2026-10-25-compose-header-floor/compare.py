"""Old (25e6818 / ffb5cc7) vs current compose.py: are the EOSTables bit-identical?

For each input, np.array_equal on every EOSTable field (n, eps, p, h, cs2, and the
MeV/fm^3 copies) between the old eos_table_from_arrays (copied verbatim below) and
the current one; where they differ, M_max, R_1.4, Lambda_1.4 of both (60-star
family) and a direct solve against the SciPy reference.  Also old vs new
read_compose on the real eos/116 and eos/188 files.  JAX: run on SLURM only
(compare.sbatch).  Writes compare.json next to this file.
docs/devlog/2026-10-25-compose-header-floor.md."""
import json, os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import nmbackend
from nmbackend.eos import RMFTEOS, MetaModelEOS
from nmbackend.eos.base import build_h_and_cs2, extend_to_surface
from nmbackend.eos.crust import sly_crust
from nmbackend.eos import compose as C
from nmbackend.priors.bridge import draw_bridge_eos
from nmbackend.tov.family import build_family, default_hc_grid, m_max, radius_of_mass, lambda_of_mass
from nmbackend.tov.solver import solve_structure
from nmbackend.reference.scipy_tov import solve_reference
print("nmbackend", nmbackend.__file__, flush=True)
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "..", "..", "tests", "data")


def old_table(n, eps, p, n_surface=1e-7, gamma_crust=1.3, xp=None):     # origin/main ffb5cc7
    n = np.asarray(n, float)
    eps = np.maximum.accumulate(np.asarray(eps, float))
    p = np.maximum.accumulate(np.maximum(np.asarray(p, float), 1e-12))
    if n_surface is not None and n[0] > n_surface:
        n, eps, p = extend_to_surface(n, eps, p, n_surface=n_surface, gamma_crust=gamma_crust)
    return build_h_and_cs2(n, eps, p)


def old_read(d):                                                         # origin/main ffb5cc7
    nb_tokens = np.loadtxt(os.path.join(d, "eos.nb"), ndmin=1)
    n_arr = nb_tokens[1:] if nb_tokens.size > 1 and float(
        nb_tokens[0]).is_integer() and int(nb_tokens[0]) == nb_tokens.size - 1 else nb_tokens
    with open(os.path.join(d, "eos.thermo")) as fh:
        header = fh.readline().split()
        m_n = float(header[0])
        rows = [list(map(float, ln.split())) for ln in fh if ln.strip()]
    rows = np.array([r for r in rows if len(r) >= 11])
    i_nb = rows[:, 1].astype(int) - 1
    n = n_arr[i_nb]
    p = rows[:, 3] * n
    eps = (rows[:, 9] + 1.0) * n * m_n
    o = np.argsort(n)
    return n[o], eps[o], p[o]


def obs(t):
    try:
        f = build_family(t, default_hc_grid(t, n_models=60))
        return dict(Mmax=float(m_max(f)), R14=float(radius_of_mass(f, 1.4)), L14=float(lambda_of_mass(f, 1.4)))
    except Exception as e:  # noqa: BLE001 -- report, do not stop
        return dict(error=repr(e)[:200])


FIELDS = ("n", "eps", "p", "h", "cs2", "eps_mevfm3", "p_mevfm3")


def field_equal(a, b):
    """{field: np.array_equal} over the EOSTable fields (False on a shape change)."""
    return {f: bool(np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))))
            for f in FIELDS}


cases = {}
t = RMFTEOS(crust="polytrope").eos_table(); n = np.asarray(t.n); m = n > 3e-4
cases["rmft_polytrope_n>3e-4 (test_compose)"] = (n[m], np.asarray(t.eps_mevfm3)[m], np.asarray(t.p_mevfm3)[m], {})
t = RMFTEOS().eos_table()
A = (np.asarray(t.n), np.asarray(t.eps_mevfm3), np.asarray(t.p_mevfm3))
cases["rmft_sly_full (test_compose roundtrip)"] = (*A, {})
m = A[0] > 3e-4
cases["rmft_sly_n>3e-4 (test_model_diagnostic)"] = (A[0][m], A[1][m], A[2][m], {})
t = MetaModelEOS().eos_table()
cases["metamodel_default"] = (np.asarray(t.n), np.asarray(t.eps_mevfm3), np.asarray(t.p_mevfm3), {})
for seed in (1, 11):
    d = draw_bridge_eos(np.random.default_rng(seed), corr_frac=0.3)
    cases[f"bridge_seed{seed}"] = (d["n"], d["eps"], d["p"], dict(n_surface=None))
ns, es, ps = sly_crust(); k = ns < A[0][0]
cases["rmft_sly + raw SLy surface rows"] = (np.r_[ns[k], A[0]], np.r_[es[k], A[1]], np.r_[ps[k], A[2]], dict(n_surface=None))
for e in ("116", "188"):
    nn, ee, pp, _ = C.read_compose(os.path.join(D, f"compose_eos{e}"))
    cases[f"eos{e} (new reader)"] = (nn, ee, pp, {})

out = {}
for name, (n, e, p, kw) in cases.items():
    to, tn = old_table(n, e, p, **kw), C.eos_table_from_arrays(n, e, p, **kw)
    r = dict(rows=int(len(n)), p_min=float(np.min(p)), rows_p_le_1e12=int(np.sum(np.asarray(p) <= 1e-12)),
             equal=field_equal(to, tn))
    r["identical"] = all(r["equal"].values())
    if not r["identical"]:
        r["old"], r["new"] = obs(to), obs(tn)
        hc = 0.6 * float(tn.h[-1])
        a = solve_structure(tn, hc, tidal=True, rtol=1e-10, atol=1e-12)
        b = solve_reference(tn, hc, rtol=1e-10, atol=1e-12)
        r["new_vs_ref_at_0.6hmax"] = dict(R=float(a.R_km), R_ref=float(b.R_km), L=float(a.Lambda), L_ref=float(b.Lambda))
    else:
        r["new"] = obs(tn)
    out[name] = r
    print(name, json.dumps(r), flush=True)

for e in ("116", "188"):
    d = os.path.join(D, f"compose_eos{e}")
    no, eo, po = old_read(d)
    nn, en, pn, _ = C.read_compose(d)
    r = dict(max_n_ratio=float(np.max(np.maximum(no / nn, nn / no))), old_n_first=no[:3].tolist(), new_n_first=nn[:3].tolist())
    r["old_reader"] = obs(C.eos_table_from_arrays(no, eo, po))
    r["new_reader"] = obs(C.eos_table_from_arrays(nn, en, pn))
    out[f"reader_eos{e}"] = r
    print(f"reader_eos{e}", json.dumps(r), flush=True)
import subprocess
out["provenance"] = dict(nmbackend=nmbackend.__file__, job=os.environ.get("SLURM_JOB_ID"),
                         head=subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True,
                                             text=True).stdout.strip(),
                         dirty=subprocess.run(["git", "-C", HERE, "status", "--porcelain", "--", ":/src", ":/tests"],
                                              capture_output=True, text=True).stdout.strip())
json.dump(out, open(os.path.join(HERE, "compare.json"), "w"), indent=1)
