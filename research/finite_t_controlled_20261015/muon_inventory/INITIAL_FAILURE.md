# Preserved driver-only initial failure

The first contract-probe invocation completed the physical calculations but failed JSON serialization: NumPy comparison returned `numpy.bool_`, which standard `json.dumps` cannot encode (displayed TypeError: Object of type bool is not JSON serializable). No results file was written. The driver now explicitly converts the check result to Python `bool`; no EOS or reaction changes. This is a reporting bug, not an EOS or weak-rate qualification failure.
