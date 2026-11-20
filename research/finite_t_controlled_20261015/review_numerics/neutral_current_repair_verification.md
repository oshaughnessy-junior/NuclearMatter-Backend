# Independent verification: neutral-current input and pair-accumulation repairs

15 October 2026. **Accept the repaired selected-case numerical contract.** `neutral_current_repair_verification.json` records **22 passing checks** on repaired `free_nc.py`, SHA256 `b802b1799188dfb929ff44f2094db3e578009865400a5a548595f9f2c48d126c`. Reproduce with `python3 outputs/nmb-finite-t-20261015/review_numerics/verify_nc_repair.py`; initial evidence is preserved.

All four original counterexamples now raise ValueError: negative moment tail, negative radiation temperature, NaN grid occupation and NaN grid weight. Array finite/positivity checks run before collision accumulation; moments validate finite positive controls and a positive radiation temperature. The initial source/results and this review's initial failures remain preserved separately.

Repeating all 15 independent valid-case checks continues to pass. The valid finite-M/heavy-M moment data are **exactly unchanged** by the repair. The independent q/omega integration, reverse-DOS KMS relation, irregular-grid number/FD equilibrium and bath-energy event sums remain accepted within the selected model.

The off-diagonal unordered pair accumulation removes large diagonal self-transitions before subtracting their contribution to the redistribution source. Those transitions should cancel identically. Independent adaptive-angle event rates reproduce the repaired held-out irregular-grid entropy and bath-energy regression. The macroscopic entropy production agrees with the microscopic sum of `(forward-backward) log(forward/backward)` to 4.22e-15 relative. This tests the fixed physical kernel while avoiding the original row-minus-column cancellation.

This supersedes the resolved input-contract rejection in `neutral_current_review.md`; it does not remove model limitations or convert structural closed-grid tests into continuum transport-rate validation. The bath remains fixed, and the nonrelativistic leading-current model still needs separately qualified extensions for interactions, physical current corrections and continuum radiation evolution.

No author files changed in this review.
