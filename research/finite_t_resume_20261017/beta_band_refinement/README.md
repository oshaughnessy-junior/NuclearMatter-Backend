# Selected upper beta interval refinement

Author-frozen research study of **only [1.1,1.4] MeV**, using exact actual evaluations of the frozen accepted conditional beta kernel. No source, current, bath, masses or occupations were changed. `LOCK.json` preceded every physical output. `REPORT.md` gives the equations, evidence and limits.

Exactly 72 physical calls were made: 64 incident nodes at Gauss orders 12/20/32 plus eight isolated/alternate-radial edge controls. All raw nodes remain separate in `raw/`. No fits, interpolation, moment substitution or zero filling were used.

**Six positive-branch order12→20 failures remain preserved.** Their errors are about 1.8–2.0e−5. The prelocked order20→32 comparison passes every positive number/energy branch at about 6e−14. The author status therefore admits only the selected higher-order finite-interval comparison, pending DIFFERENT review. It does not rehabilitate order12 or close the prior failed calculation.

`analyze.py` is saved-only arithmetic and can be rerun without new physics calls. `run_batch.py` refuses raw-result overwrites; do not run extra profiles without root authorization. Every physical batch used one numerical thread and a fresh account/window/reset/quota check before dispatch.

Phi-axis isolation, rigorous uniform tail bounds, other incident bands, full moments, event-total parity and transport evolution remain OPEN. The empty radiation backward source is exactly absent incoming occupation; true chi is positive at every node and is retained separately.
