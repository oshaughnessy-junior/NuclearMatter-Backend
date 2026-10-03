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


@pytest.fixture(scope="session")
def nscool_probe(tmp_path_factory):
    """Call single NSCool routines (tests/nscool_probe.f) on rows (key, a, b, c, d).

    Links the probe against the NSCool build's objects; skips without
    gfortran or a build unless NSCOOL_REQUIRED is set.
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path

    import numpy as np

    from nmbackend.cooling import default_home

    code = default_home() / "Code"
    objs = [code / f"{m}.o" for m in ("neutrino_core", "neutrino_crust", "spec_heat", "Tools")]
    fc = os.environ.get("FC", "gfortran")
    if not (shutil.which(fc) and all(o.is_file() for o in objs)):
        if os.environ.get("NSCOOL_REQUIRED"):
            raise RuntimeError("NSCool objects or gfortran missing")
        pytest.skip("NSCool build or gfortran missing")
    exe = tmp_path_factory.mktemp("probe") / "probe"
    subprocess.run([fc, "-O0", "-std=legacy", "-w", "-fno-automatic", "-o", str(exe),
                    str(Path(__file__).with_name("nscool_probe.f"))] + [str(o) for o in objs],
                   check=True)

    def run(rows):
        text = "".join(f"{k} {a!r} {b!r} {c!r} {d!r}\n" for k, a, b, c, d in rows)
        out = subprocess.run([str(exe)], input=text, capture_output=True, text=True,
                             cwd=code.parent, check=True).stdout.split()
        return np.array([float(x) for x in out])

    return run
