# Disposable cross-host transport fixture

This directory is an isolated review-system canary. It is not an NMB physics
implementation and must never be merged. No production package imports it.

Contract: `weighted_mean(values, weights)` accepts equal-length nonempty lists
of finite numbers with strictly positive finite weights. Return sum(v*w)/sum(w).
Multiplying every weight by the same positive constant must preserve the result.
The focused standard-library test is `python -m unittest discover -s .review-canary -p "test_*.py"`.
