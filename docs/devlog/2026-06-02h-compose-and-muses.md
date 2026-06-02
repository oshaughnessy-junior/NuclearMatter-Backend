# Dev log — 2026-06-02 (h) — CompOSE ingestion, model diagnostic, MUSES findings

**Who:** R. O'Shaughnessy (with AI pair).  Phase 2 cross-check.

## CompOSE / tabular ingestion (done)
`eos.compose`: read CompOSE `eos.nb`+`eos.thermo` (p=Q1 n_B, eps=(Q7+1) n_B m_n;
Q1 col 3, Q7 col 9 — fixed an off-by-one in the reader), a simple 3-column table,
and a MUSES CMF CSV (n_B col 5, eps col 8, p col 9) into the standard `EOSTable`
(`load_compose`/`load_simple`/`load_cmf`/`TabularEOS`); `write_compose` exports.
Validated by **round-trip**: RMFT(GM1) -> CompOSE files -> table -> TOV reproduces
M_max/R/Lambda identically (~1e-11), and the ingested table cross-codes (diffrax vs
SciPy) at the internal floor. So the thermodynamically consistent h(mu) interpolation
handles real tabular EOS as well as the parametric models.

## EOS-level cross-code diagnostic (done)
`validation.model_diagnostic` compares two EOS at the THERMODYNAMIC level — eps, p,
mu=(eps+p)/n, cs2=dp/deps vs density — not just M-R-Lambda. RMFT vs its CompOSE/CMF
round-trip = 0 to machine precision; RMFT vs metamodel differs up to 89% in p, 110%
in cs2 (real model difference TOV would partly hide). This is the "model, not TOV"
diagnostic, and the natural cross-check substrate for an external code's table.

## MUSES (researched; docs/muses-integration.md)
The MUSES "RMF code" (cmf_solver) is the Dexheimer-Schramm **Chiral Mean Field**
model — non-linear SU(3) sigma model (sigma,zeta,delta,omega,phi,rho + hyperons +
quarks + Polyakov loop), grand-canonical in (mu_B,mu_S,mu_Q), C++/MINPACK. NOT a
sigma-omega-rho Walecka RMF -> its couplings don't map to our (b,c) and don't fix
NL3 (separate task); reimplementing it is unwarranted. Right path = external EOS
provider: CMF emits CompOSE CSV -> `load_cmf` -> `model_diagnostic` + TOV (no
reimplementation). Documented the model, couplings (for reference), and how to get a
real table (Docker/API + Lepton/Synthesis for the NS line).

## Tests: 40 total (added compose 5, model_diagnostic 2).

## Next (when a real CMF/qlimr table is in hand)
Run `load_cmf` + `model_diagnostic` + TOV on an actual CMF NS EOS; same for qlimr.
Separately: verify a standard linear-RMF set's couplings vs its saturation point to
add a validated second RMFTEOS parameter set.
