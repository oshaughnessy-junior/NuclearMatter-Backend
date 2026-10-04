"""Tabulated NSCool crust thermal chemistry, distinct from the native EOS density."""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class CrustTable(NamedTuple):
    """Ascending density nodes; rho in g/cm^3 and chemical n in fm^-3."""
    rho: jnp.ndarray
    n: jnp.ndarray
    A_cell: jnp.ndarray
    A_ion: jnp.ndarray
    Z: jnp.ndarray

    @classmethod
    def read(cls, path):
        """Read and validate an NSCool six-column crust file outside JIT."""
        lines = Path(path).read_text().splitlines()
        n_text, n_rows = map(int, lines[0].split()[:2])
        records = [line for line in lines[1+n_text:] if line.strip()][:n_rows]
        # Fortran reads six quantities; trailing phase-transition annotations are ignored.
        data = np.asarray([[float(v) for v in line.split()[:6]] for line in records])
        if data.shape != (n_rows, 6) or n_rows < 2 or not np.all(np.isfinite(data)):
            raise ValueError('Crust file must contain finite six-column rows')
        data = data[np.argsort(data[:, 0])]
        rho, pressure, n, cell, ion, z = data.T
        if (np.any(rho <= 0) or np.any(pressure <= 0) or np.any(n <= 0)
                or np.any(np.diff(rho) <= 0) or np.any(np.diff(n) <= 0)
                or np.any(z <= 0) or np.any(ion < z) or np.any(cell < ion)):
            raise ValueError('Invalid crust density or nuclear composition')
        return cls(*(jnp.asarray(v) for v in (rho, n, cell, ion, z)))


class CrustChemistry(NamedTuple):
    """Reference chemical n, fractions and nuclei; n does not overwrite EOS n."""
    n: jnp.ndarray
    Ye: jnp.ndarray
    Yn: jnp.ndarray
    A_cell: jnp.ndarray
    A_ion: jnp.ndarray
    Z: jnp.ndarray
    v_ion: jnp.ndarray


def crust_chemistry(table: CrustTable, rho, *, core_rho, core_n, core_Ye):
    """Thermal chemistry with a moving core-matched upper node; invalid inputs give NaN.

    Mirrors ``get_crust_chemistry`` in NSCool ``precool.f``. Intervals are
    lower < rho <= upper. Derivatives are conditional between species/join
    contacts. Queries must exceed the lowest node and not exceed core_rho.
    The supplied core electron fraction sets the virtual top's A_cell.
    """
    last = jnp.clip(jnp.searchsorted(table.rho, core_rho, side='left')-1,
                    0, table.rho.size-1)
    upper = jnp.clip(jnp.searchsorted(table.rho, rho, side='left'), 1, last+1)
    lower = upper-1
    index = jnp.minimum(upper, table.rho.size-1)
    virtual = upper == last+1
    safe_ye = jnp.where(core_Ye > 0, core_Ye, 1.)
    top_cell = table.Z[last]/safe_ye
    hi_rho = jnp.where(virtual, core_rho, table.rho[index])
    hi_n = jnp.where(virtual, core_n, table.n[index])
    hi_cell = jnp.where(virtual, top_cell, table.A_cell[index])
    hi_ion = jnp.where(virtual, table.A_ion[last], table.A_ion[index])
    hi_z = jnp.where(virtual, table.Z[last], table.Z[index])
    width = hi_rho-table.rho[lower]
    weight = (rho-table.rho[lower])/jnp.where(width > 0, width, 1.)
    cell = jnp.where(hi_cell == hi_ion, hi_cell,
                     table.A_cell[lower]+weight*(hi_cell-table.A_cell[lower]))
    n = table.n[lower]+weight*(hi_n-table.n[lower])
    safe_cell = jnp.where(cell > 0, cell, 1.)
    ye, yn = hi_z/safe_cell, (cell-hi_ion)/safe_cell
    # The nuclear mass cancels in NSCool's n_ion * v_nucleus prescription.
    volume = jnp.minimum(1., rho*(4./3.*3.14159*1.1**3*1e-39)/(1.66e-24))
    values = CrustChemistry(n, ye, yn, cell, hi_ion, hi_z, volume)
    valid = (jnp.isfinite(rho) & jnp.isfinite(core_rho) & jnp.isfinite(core_n)
             & jnp.isfinite(core_Ye) & (core_rho > table.rho[0]) & (core_n > 0)
             & (core_Ye > 0) & (core_Ye <= 1)
             & jnp.isfinite(top_cell) & jnp.isfinite(cell) & jnp.isfinite(n)
             & (top_cell >= table.A_ion[last]) & (core_n >= table.n[last])
             & (rho > table.rho[0])
             & (rho <= core_rho) & (width > 0) & (n > 0)
             & (hi_z > 0) & (hi_ion >= hi_z) & (cell >= hi_ion))
    return CrustChemistry(*(jnp.where(valid, value, jnp.nan) for value in values))


def _unpaired_crust_terms(T_local, rho, chemistry, *, rho_drip):
    """Local unpaired crust cv and neutrino emissivity, without GR weights.

    Mirrors NSCool's EMNCR=5 neutron mass prescription and unpaired thermal
    calls. Chemical density is distinct from EOS energy density. The caller
    owns density coverage and moving species, drip and ionic phase contacts.
    The unsupported ionic source-table overrun branch must be excluded.
    """
    from . import crust_nu, heat
    from .native import _kf

    ke = _kf(chemistry.n, chemistry.Ye)
    kn = _kf(chemistry.n, chemistry.Yn)
    mn = jnp.minimum(1., 1.09-.11*kn)
    inner = rho >= rho_drip
    cv = (heat.lepton_cv_coeff(ke, .511)*T_local
          + heat.cv_ion(T_local, rho, chemistry.A_cell, chemistry.Z)
          + jnp.where(inner, heat.cv_coeff(kn, 939.56*mn)*T_local, 0.))
    q = (crust_nu.eebrem(T_local, ke) + crust_nu.npb(T_local, rho)
         + crust_nu.plasma(T_local, rho, chemistry.A_cell, chemistry.Z)
         + jnp.where(inner, crust_nu.brem_nn_crust(
             T_local, kn, mn, 0., chemistry.v_ion, 0, -1), 0.))
    return cv, q
