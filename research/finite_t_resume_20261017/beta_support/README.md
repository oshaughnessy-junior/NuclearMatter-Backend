# Upper beta spectral support checkpoint

`REPORT.md` derives exact parent-energy/cone and cluster-threshold maps, preserves the original local tail failure, and states the spin/DOS normalization. `conditional_beta.py` is a separate research candidate; `beta_pr55_reference/` contains exact source copies under their explicit namespace. No runtime aliases or original-source mutations are used.

Eight profile calls are complete. Their immutable raw JSON files are `OLD_TAIL70`, `OLD_TAIL90`, and `E120/E130_PANEL/SEMI16/SEMI24`. To avoid accidental budget overrun, do not rerun those profiles without a new root-authorized experiment. The driver refuses to overwrite raw result names. `phase_jacobian.py` is an unintegrated coordinate check; `assess.py` reads existing results and verifies source identities without new physical integrations.

Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for any separately authorized execution. No dependencies were installed. Frozen results require DIFFERENT review; coupled profile agreement is not an all-axis, whole-energy or transport qualification.
