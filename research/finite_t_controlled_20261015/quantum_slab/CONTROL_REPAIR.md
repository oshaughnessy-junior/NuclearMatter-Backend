# Independent-review control-contract repair

Independent numerical review accepted17 valid held-out slab/guard checks but found that a broad `ValueError` fallback accepted a negative integration tail if the unrelated float64 collapsed-panel witness was also present. Exact failing arguments: moderately FD matter T=2MeV,nn=.002,np=.001, nue E=60MeV,order72,tail=-1. The review preserves its original failure report; `initial_fixed_bath.py` and `initial_guarded_capture.py` preserve the executable pre-repair source.

The active guard now validates the same finite-positive energy/tail and integer-order controls as mapped capture, both directly and at slab entry. The fallback only handles the original `math domain error` and still requires a recorded collapsed panel; other exceptions propagate. Nine targeted control tests pass, including the negative-tail counterexample and valid positive-tail panel witness. The parent reran the full506-check slab and8-check guard suites successfully after this change.

`CHECKPOINT.json` remains the original author's historical manifest. `initial_author_validation_results.json` preserves that original report. `REPAIRED_SOURCE_HASHES.json` supplements it with the current source/data hashes. Independent post-repair review establishes acceptance; no broader domain or spectrum qualification follows from this input-control repair.
