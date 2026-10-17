# Scoped subnormal-input admission repair

The independent initial review in `review_pairing_pbf_contract/` remains frozen at92PASS,2 new subnormal-input FAIL and4 known raw legacy endpoint FAIL. The original author tree, including source, raw results, manifest and the earlier helper failure, is copied byte-for-byte to `pre_admission_repair/`; `PRE_ADMISSION_REPAIR_PIN.json` records all28 file hashes. The reviewer also preserves the initial author tree independently. No reviewer file was edited.

Both `log_F_A` and `log_F_B` now require **either exact zero or positive normal float64 input**, subject to their unchanged previous upper bounds of1000 and500 respectively. Specifically,0<v<np.finfo(float).tiny is rejected with ValueError. Tiny is2.2250738585072014e−308; it and its next representable neighbor above remain allowed. Exactly zero, including negative zero, retains the prior log−infinity/exact-zero representation. All other pre-existing domain policies remain unchanged.

The observed accuracy failure is angular rounding in FB: multiplying a subnormal v by sqrt(1+3mu²) loses physical input variation before log evaluation. FA is given the same conservative input admission; this is not a claim that the review found the same FA error. No integrand, angular factor, quadrature tolerance, weak-current prescription or legacy scientific function was altered.

Input exclusion is distinct from ordinary **output** underflow. A normal input may produce a positive function far below float64 representation. Those outputs still return a finite log, ordinary value None and `positive_below_float64_representation`; no floor or artificial exact zero is introduced. Smallest-normal inputs, v=1e−200 and the large-gap deep-output examples remain supported. Rejecting the demonstrated subnormal extension does not prove uniform accuracy over every remaining normal argument.

`check_admission_repair.py` has **66 passing targeted controls**, no failed repair gate, and ran in.74s:

- Replays the exact old1e−320 and nextafter(0,+1) angular errors at the unchanged3e−10 comparison gate, preserving the old values/warnings, then verifies rejection by the repaired helper.
- Checks the largest subnormal value immediately below tiny rejects, while tiny and nextafter(tiny,+infinity) satisfy the original small-gap analytic coefficient gate.
- Confirms bitwise identity of both logs at13 selected normal/zero inputs, including the smallest normal,1e−307,1e−250,1e−200 and ordinary physical gaps; the two large-gap unrepresented examples are also bitwise unchanged.
- Verifies exact-zero/invalid-input policies and positive-log/None output statuses remain unchanged.
- Checks normalized scientific AST identity after removing only the two new admission guards and changed docstrings. Exact pinned NMB functions, source snapshots, original science reports, Fortran comparisons, endpoint reports and the previous helper failure all retain their original hashes.

The four raw NMB T0/Tc endpoint failures remain exactly as originally reported. There is no legacy T0/Tc fix, new cooling/paired-EOS/AD qualification or wider log-domain accuracy certificate. The original132 selected author checks and initial independent92/2/4 record remain separate preserved artifacts; this repair report does not rewrite their counts or turn their failures into passes. Separate recheck by the original independent reviewer is pending.
