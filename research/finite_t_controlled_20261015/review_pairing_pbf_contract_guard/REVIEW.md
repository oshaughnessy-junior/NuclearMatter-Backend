# Independent PBF subnormal-input admission recheck

**ACCEPTED for the narrow helper input-admission repair.** There are **251 passing repair/provenance/compatibility checks, no new failed repair gate**, and **two freshly reproduced original FB accuracy failures retained explicitly as FAIL**. Many passing checks are file hashes or repeated normal inputs, not independent physical benchmarks. No broader PBF qualification follows.

Reviewed `pairing_pbf_contract/quasiparticle.py` SHA256 `06f6db3fb69f2fe50042a15d5c129f151354561c3263e4cc976e7cf869a419d0` and current `MANIFEST.json` SHA256 `56b2ac889f605bef332c5a59894aa4f7fca5100521bfc5ad5511e58be2294a7f`. I read the admission-repair note, preserved pre-repair source/manifest/results and prior independent review before executing the new probe. Every current manifested file and all 28 pre-repair pinned files verify. The separate prior review's archived initial author copies also verify against those 28 pins. All 111 files under the author and prior-review trees remain byte-exact after the recheck. No author script/driver was executed.

The old helper is imported directly from its frozen `pre_admission_repair/` copy in a fresh process. A separately evaluated scalar direct-x integral supplies the small-gap coefficient; it agrees with 24[eta(5)−eta(4)] at the unchanged selected gate. Comparing old FB with 2log(v)+log(2*coefficient) freshly reproduces:

| Old positive input | Relative FB error | Old 3e−10 gate | Repaired admission |
|---|---:|---|---|
| 1e−320 | 3.917239779050859e−6 | FAIL | ValueError |
| nextafter(0,+1)=4.9406564584124654e−324 | 0.031754163466144236 | FAIL | ValueError |

The first old calculation emits the original maximum-120-subdivisions QUADPACK warning, retained verbatim in `results.json`. The old failures remain failed numerical measurements; reproducing them is separately counted as a successful diagnosis. Both repaired FA and FB reject these inputs. This does not claim the old FA had the same angular defect.

The precise float64 boundary is checked using adjacent representable values. `nextafter(tiny,0)` (largest subnormal, hexadecimal `0x0.fffffffffffffp-1022`) rejects in both functions; `tiny=2.2250738585072014e−308` (`0x1.0000000000000p-1022`) and `nextafter(tiny,+infinity)` (`0x1.0000000000001p-1022`) remain admitted and recover both analytic small-gap coefficients within the unchanged 3e−10 relative gate. These tests qualify the selected boundary, not every normal argument.

For each function I compare old/new float-hex log values and representation dictionaries at 18 inputs: +0, −0, tiny, its next higher value, 1e−307, 1e−280, 1e−250, 1e−200, 1e−163, .017, .43, 2.7, 9.3, 37, 127, 300, 499 and 500. All are bitwise/dictionary identical. FA at 1000 is also unchanged. Both signs of exact zero return log−infinity and the unchanged `exact_zero` representation. Selected malformed/nonfinite/negative inputs and the first representable argument above each existing upper bound remain rejected.

Normal small-gap and deep large-gap inputs continue to produce finite logs with `value=None` and `positive_below_float64_representation` when the **output** underflows. In particular FA(1000), FB(500), and both helpers at 1e−200 and tiny retain this behavior. No floor or artificial exact zero was introduced. Input exclusion and output representation are distinct policies.

A literal whole-module AST comparison is identical after removing only the two exact new admission guards and function docstrings. No integrand, physical angular factor, weak prescription, quadrature tolerance or representation function changes. This structural check and the selected bitwise comparisons support a compatibility claim within their stated scope; they do not prove uniform physical accuracy.

The initial independent **92 PASS / 2 FAIL** record and its exact two failed subnormal extension artifacts remain untouched. The **four known raw legacy T0/Tc endpoint failures** and all their source/evidence bytes remain untouched and excluded. They were not newly rerun or repaired in this narrow admission recheck. Earlier helper failures and original author science reports likewise remain preserved. This acceptance adds no conserving weak tensor, paired EOS, heat-capacity/entropy/condensation-energy consistency, cooling trajectory, opacity, derivative/AD or modern physical uncertainty qualification.

Reproduce from `/Users/rossma/Genesis_LANL`:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_pairing_pbf_contract_guard/probe.py
```

The complete experiment took .397 s, below the 120 s bound. `results.json` holds every gate, old failure/warning, exact input hexadecimal boundary and selected old/new log hexadecimal values. `before.json` and `CHECKPOINT.json` pin frozen provenance and this review. Fresh account/window verification at 2026-10-16 00:16:02 UTC matched account `82f4cfd9-9e20-47cf-9997-761328d89f60`, 10080-minute reset 1792652325, 22% used. No reset, install, build, production, paper or Git action occurred.
