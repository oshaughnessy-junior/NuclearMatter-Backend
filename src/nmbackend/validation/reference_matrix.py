"""Reference-matrix harness: run several TOV solvers on a ladder of EOS and tabulate
per-quantity differences **at fixed gravitational mass**.

Comparing at fixed mass (not fixed central density/enthalpy) is deliberate: it is
convention-independent, so RePrimAnd (central gm1), our solver (central h), and any
other reference are compared on the physically meaningful M-R-Lambda relation.

Solvers (each is *guarded* -- absent backends are simply skipped):
    diffrax    : nmbackend production adaptive solver
    scipy_hbeta: independent radius-domain SciPy RK45 + (H,beta) reference
    reprimand  : RePrimAnd 1.7 (external; needs pyreprimand)
    lalsim     : lalsimulation named-EOS path (planned; needs a shared table)
    qlimr      : MUSES qlimr observables (planned; external -- source not yet vendored)

Output: for each EOS and target mass, R [km] and Lambda from each solver, plus the
difference vs the chosen reference (Delta R in metres, Delta Lambda in percent).
The reference defaults to the most accurate available: reprimand > scipy_hbeta.

Lightweight by design: one file, numpy + the package's own solvers, no plotting.
Run:  PYTHONPATH=src python -m nmbackend.validation.reference_matrix
"""
from __future__ import annotations

import numpy as np

from ..eos import MetaModelEOS, HybridEOS
from ..tov import build_family, solve_hybrid
from ..tov.family import default_hc_grid
from ..reference.scipy_tov import solve_reference, solve_reference_hybrid

try:
    from ..reference import reprimand as _rpa
    _HAVE_RPA = bool(_rpa.HAVE_REPRIMAND)
except Exception:
    _HAVE_RPA = False


# ---------- stable-branch utilities ----------
def _rising_branch(M, R, L):
    """Stable, mass-increasing branch up to M_max, sorted by mass."""
    M, R, L = map(lambda a: np.asarray(a, float), (M, R, L))
    ok = np.isfinite(M) & np.isfinite(R) & (M > 0)
    M, R, L = M[ok], R[ok], L[ok]
    if M.size < 3:
        return M, R, L
    imax = int(np.argmax(M))
    M, R, L = M[:imax + 1], R[:imax + 1], L[:imax + 1]
    o = np.argsort(M)
    return M[o], R[o], L[o]


def _at_masses(M, R, L, targets):
    """Interpolate R, Lambda at target masses (NaN outside the branch / above Mmax)."""
    M, R, L = _rising_branch(M, R, L)
    if M.size < 2:
        nan = np.full(len(targets), np.nan)
        return nan, nan, (M.max() if M.size else np.nan)
    Rt = np.interp(targets, M, R, left=np.nan, right=np.nan)
    Lt = np.interp(targets, M, L, left=np.nan, right=np.nan)
    return Rt, Lt, float(M.max())


# ---------- solver sequence runners (return raw M, R, Lambda arrays) ----------
def _seq_diffrax_smooth(table, hc):
    f = build_family(table, hc)
    return np.asarray(f.M_msun), np.asarray(f.R_km), np.asarray(f.Lambda)


def _seq_scipy_smooth(table, hc):
    pts = [solve_reference(table, float(h)) for h in hc]
    return (np.array([p.M_msun for p in pts]),
            np.array([p.R_km for p in pts]),
            np.array([p.Lambda for p in pts]))


def _seq_rpa_smooth(table, hc):
    M, R, L, _ = _rpa.reprimand_sequence(table)
    return np.asarray(M), np.asarray(R), np.asarray(L)


def _seq_diffrax_hybrid(hybrid, hc):
    pts = [solve_hybrid(hybrid, float(h)) for h in hc]
    return (np.array([float(p.M_msun) for p in pts]),
            np.array([float(p.R_km) for p in pts]),
            np.array([float(p.Lambda) for p in pts]))


def _seq_scipy_hybrid(hybrid, hc):
    pts = [solve_reference_hybrid(hybrid, float(h)) for h in hc]
    return (np.array([p.M_msun for p in pts]),
            np.array([p.R_km for p in pts]),
            np.array([p.Lambda for p in pts]))


def _seq_rpa_hybrid(hybrid, hc):
    M, R, L, _ = _rpa.reprimand_sequence(hybrid.eos_table())
    return np.asarray(M), np.asarray(R), np.asarray(L)


# ---------- EOS ladder ----------
def build_ladder():
    """(name, kind, obj, hc_grid_fn) for each EOS in the adversarial ladder."""
    mm = MetaModelEOS()
    cases = [
        ("MetaModel(L60, cs2_tail=0.4)", "smooth", mm.eos_table(), None),
        ("MetaModel(stiff cs2_tail=0.6)", "smooth", mm.eos_table({"cs2_tail": 0.6}), None),
        ("MetaModel(soft  cs2_tail=0.3)", "smooth", mm.eos_table({"cs2_tail": 0.3}), None),
        ("MetaModel(L40)", "smooth", mm.eos_table({"L_sym": 40.0}), None),
        ("Hybrid(pt=80, de=250, cs2q=1)", "hybrid",
         HybridEOS(p_trans_mev=80.0, delta_eps_mev=250.0, cs2_quark=1.0), None),
    ]
    return cases


# ---------- runners per kind ----------
def _runners(kind):
    if kind == "smooth":
        r = {"diffrax": _seq_diffrax_smooth, "scipy_hbeta": _seq_scipy_smooth}
        if _HAVE_RPA:
            r["reprimand"] = _seq_rpa_smooth
    else:  # hybrid
        r = {"diffrax": _seq_diffrax_hybrid, "scipy_hbeta": _seq_scipy_hybrid}
        if _HAVE_RPA:
            r["reprimand"] = _seq_rpa_hybrid
    return r


def _hc_grid(kind, obj, n_models):
    if kind == "smooth":
        return default_hc_grid(obj, n_models=n_models)
    table = obj.eos_table()
    return np.linspace(0.04 * float(table.h[-1]), 0.92 * float(table.h[-1]), n_models)


# ---------- main comparison ----------
def run(target_masses=(1.2, 1.4, 1.6, 1.8, 2.0), n_models=60, reference=None,
        csv_path=None):
    ladder = build_ladder()
    tm = np.asarray(target_masses, float)
    rows = []   # long-format records for optional CSV (track vs budget over time)

    print(f"Reference-matrix harness | targets M[Msun]={list(tm)} | n_models={n_models}")
    print(f"available solvers: diffrax, scipy_hbeta"
          + (", reprimand" if _HAVE_RPA else "  (reprimand: not installed)"))
    print("Delta R in metres, Delta Lambda in percent, vs the reference column.\n")

    for name, kind, obj, _ in ladder:
        runners = _runners(kind)
        hc = _hc_grid(kind, obj, n_models)
        seqs = {}
        for sname, fn in runners.items():
            try:
                seqs[sname] = _at_masses(*fn(obj, hc), tm)
            except Exception as e:
                print(f"  [{sname} failed on {name}: {e!r}]")
        ref = reference or ("reprimand" if "reprimand" in seqs else "scipy_hbeta")
        if ref not in seqs:
            print(f"=== {name}: reference {ref} unavailable, skipping ===\n")
            continue
        Rref, Lref, Mmax_ref = seqs[ref]
        order = [s for s in ("reprimand", "scipy_hbeta", "diffrax") if s in seqs]

        print(f"=== {name}   (reference = {ref}) ===")
        hdr = "  M    " + "".join(f"{'R_'+s:>12}" for s in order) \
              + "".join(f"{'dR_'+s+'[m]':>16}" for s in order if s != ref)
        print(hdr)
        for i, M in enumerate(tm):
            row = f"  {M:4.2f} "
            for s in order:
                row += f"{seqs[s][0][i]:12.4f}"
            for s in order:
                if s == ref:
                    continue
                dR_m = (seqs[s][0][i] - Rref[i]) * 1000.0   # km -> m
                row += f"{dR_m:16.1f}"
            print(row)
        hdrL = "  M    " + "".join(f"{'L_'+s:>12}" for s in order) \
               + "".join(f"{'dL_'+s+'[%]':>14}" for s in order if s != ref)
        print(hdrL)
        for i, M in enumerate(tm):
            row = f"  {M:4.2f} "
            for s in order:
                row += f"{seqs[s][1][i]:12.2f}"
            for s in order:
                if s == ref:
                    continue
                dL = 100.0 * (seqs[s][1][i] - Lref[i]) / Lref[i]
                row += f"{dL:14.2f}"
            print(row)
        print(f"  (M_max[{ref}] = {Mmax_ref:.3f} Msun)\n")

        for i, M in enumerate(tm):
            for s in order:
                rows.append(dict(
                    eos=name, M_target=float(M), solver=s, reference=ref,
                    R_km=float(seqs[s][0][i]), Lambda=float(seqs[s][1][i]),
                    dR_m=float((seqs[s][0][i] - Rref[i]) * 1000.0),
                    dLambda_pct=float(100.0 * (seqs[s][1][i] - Lref[i]) / Lref[i]),
                ))

    if csv_path:
        import csv
        cols = ["eos", "M_target", "solver", "reference", "R_km", "Lambda",
                "dR_m", "dLambda_pct"]
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {csv_path}")
    return rows


if __name__ == "__main__":
    run()
