# Locked compact capture fit: admission blocker before physical collection

The 12 training and8 heldout states,7 basis functions and1% independent per-output heldout gates are frozen in LOCK.json (SHA256 e5b97f0023a02b8f15d036765082499b6d9705f1037ac75bcc3d3a099dfaf5cd). No detailed training/heldout source data or coefficients have been produced yet.

EOS preflight has actual matched roots for all20 states. Exactly one training state, T=1MeV,nB=1e-6fm^-3,Ye=.5, has total mu_e/T=4.197548892941277. The existing EC event function explicitly admits only |mu_e/T|<=3. Its baryon fugacity controls pass and density root residuals are below3.51e-14, so this is a declared kernel-domain blocker, not a failed EOS root. All other states pass these input controls.

The collector is staged but refuses any physical collection while all_admitted is false. No source guard is relaxed, no out-of-contract convenience API is called and no training state or basis is silently changed. A separately authorized and independently qualified physical admission extension is required for the complete locked experiment. The original physical chain remains default/reference; any eventual fit is an optional correction to the named allowed1D model, not a net-source solver or AD-chain substitute.
