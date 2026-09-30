import jax
import pytest


def _n_maps():
    try:
        with open("/proc/self/maps") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


@pytest.fixture(autouse=True, scope="module")
def _bound_jax_caches():
    # Every cached XLA executable holds ~100 memory maps; one pytest process over the
    # whole suite can reach vm.max_map_count (65530 on Linux) and abort in compile.
    yield
    if _n_maps() > 32768:
        jax.clear_caches()
