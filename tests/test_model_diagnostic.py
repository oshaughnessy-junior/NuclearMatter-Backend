"""EOS-level (model) cross-code diagnostic."""
import numpy as np
import tempfile

from nmbackend.eos import RMFTEOS, MetaModelEOS
from nmbackend.eos.compose import write_compose, load_compose
from nmbackend.validation.model_diagnostic import model_diagnostic


def test_roundtrip_is_zero_at_model_level():
    rm = RMFTEOS()
    t0 = rm.eos_table()
    n = np.asarray(t0.n)
    m = n > 3e-4
    d = tempfile.mkdtemp()
    write_compose(d, n[m], np.asarray(t0.eps_mevfm3)[m], np.asarray(t0.p_mevfm3)[m])
    t1 = load_compose(d)
    o = model_diagnostic(t0, t1, verbose=False)
    for q in ("eps", "p", "mu", "cs2"):
        assert o["maxrel"][q] < 1e-6        # identical EOS -> machine precision


def test_distinct_models_differ():
    o = model_diagnostic(RMFTEOS().eos_table(), MetaModelEOS().eos_table(),
                         verbose=False)
    # two genuinely different models must differ in the stiffness-sensitive
    # quantities (pressure, sound speed)
    assert o["maxrel"]["p"] > 0.05
    assert o["maxrel"]["cs2"] > 0.05
