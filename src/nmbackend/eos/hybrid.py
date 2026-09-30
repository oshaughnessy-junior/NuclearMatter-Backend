"""Hybrid hadron--quark EOS with a first-order (Maxwell) phase transition.

A hadronic branch (the metamodel) up to a transition pressure ``p_trans``; then a
**density discontinuity** at constant pressure (Maxwell construction) into a
constant-sound-speed (CSS) quark branch.  This is the deliberate *stress test* for
the TOV solver: across the jump ``cs^2 -> 0`` and the energy density is
discontinuous, which the smooth solver mishandles and which requires the tidal
**junction condition** (see ``nmbackend.tov.phase_transition``).

Across a Maxwell transition the pressure and the chemical potential mu = (eps+p)/n
are continuous while n and eps jump; we use mu-continuity to set the baryon density
on the quark side: n_Q = n_H (eps_Q + p_t)/(eps_H + p_t).

Two builders:
  * :meth:`segments` / :meth:`transition` -- smooth sub-tables + jump data for the
    sharp (junction-corrected) solver.
  * :meth:`eos_table` -- one combined table (jump represented by cs^2 -> 0 cell),
    for the RIFT adapter / plotting.
  * :meth:`eos_table_smeared` -- the jump replaced by a finite-width crossover, so
    the *standard* adaptive solver resolves it; used to validate the junction in the
    sharp limit (width -> 0).

Construction uses NumPy (static per parameter set); AD through the transition
parameters is future work.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ..constants import MEVFM3_TO_GEOM
from .base import build_h_and_cs2, refine_table, EOSTable
from .metamodel import MetaModelEOS


class TransitionInfo(NamedTuple):
    h_t: float        # pseudo-enthalpy at the transition (dimensionless)
    p_t: float        # transition pressure, geometric km^-2
    eps_in: float     # inner (higher-density, quark) side, geometric km^-2
    eps_out: float    # outer (lower-density, hadronic) side, geometric km^-2


class HybridEOS:
    name = "HybridEOS"

    def __init__(self, hadronic=None, had_params=None, p_trans_mev=80.0,
                 delta_eps_mev=200.0, cs2_quark=1.0, eps_max_fac=6.0,
                 n_quark=200, n_hadronic="auto"):
        self.hadronic = hadronic or MetaModelEOS()
        self.had_params = had_params
        self.p_trans_mev = p_trans_mev
        self.delta_eps_mev = delta_eps_mev
        self.cs2_quark = cs2_quark
        self.eps_max_fac = eps_max_fac
        self.n_quark = n_quark
        # node count of the hadronic segment (refine_table), which otherwise
        # varies with p_trans; "auto" is the length of the full hadronic
        # table, None keeps the raw nodes
        self.n_hadronic = n_hadronic
        self._build()

    # ---- construction ----
    def _hadronic_arrays(self):
        t = self.hadronic.eos_table(self.had_params)
        return (np.asarray(t.n), np.asarray(t.eps_mevfm3),
                np.asarray(t.p_mevfm3))

    def _build(self):
        n_h, eps_h, p_h = self._hadronic_arrays()
        pt = float(self.p_trans_mev)
        if pt <= p_h[0] or pt >= p_h[-1]:
            raise ValueError(f"p_trans={pt} outside hadronic pressure range "
                             f"[{p_h[0]:.3g}, {p_h[-1]:.3g}] MeV/fm^3")
        eps_t = float(np.interp(pt, p_h, eps_h))      # hadronic eps at p_t
        n_t = float(np.interp(pt, p_h, n_h))

        # outer (hadronic) segment: p < p_t, capped with the transition point
        m = p_h < pt
        self._had_seg = (np.append(n_h[m], n_t),
                         np.append(eps_h[m], eps_t),
                         np.append(p_h[m], pt))

        # quark (inner) CSS segment from (p_t, eps_Q)
        cs2 = float(self.cs2_quark)
        eps_Q = eps_t + float(self.delta_eps_mev)
        n_Q = n_t * (eps_Q + pt) / (eps_t + pt)        # mu-continuity
        eps_q = np.geomspace(eps_Q, eps_Q * self.eps_max_fac, self.n_quark)
        p_q = pt + cs2 * (eps_q - eps_Q)
        c0 = pt - cs2 * eps_Q
        n_q = n_Q * (((1 + cs2) * eps_q + c0) /
                     ((1 + cs2) * eps_Q + c0)) ** (1.0 / (1.0 + cs2))
        self._quark_seg = (n_q, eps_q, p_q)

        # build segment tables; quark enthalpy continues above hadronic h_t
        self._n_outer = n_h.size if self.n_hadronic == "auto" else self.n_hadronic
        if self._n_outer is None:
            outer = build_h_and_cs2(*[np.asarray(a) for a in self._had_seg])
            self._outer_raw = outer
        else:   # NumPy until the length is fixed: no compile per raw length
            self._outer_raw = build_h_and_cs2(*self._had_seg, xp=np)
            outer = refine_table(self._outer_raw, self._n_outer)
        h_t = float(outer.h[-1])
        inner = build_h_and_cs2(*[np.asarray(a) for a in self._quark_seg],
                                h_base=h_t)
        self._outer, self._inner = outer, inner
        self._transition = TransitionInfo(
            h_t=h_t, p_t=pt * MEVFM3_TO_GEOM,
            eps_in=eps_Q * MEVFM3_TO_GEOM, eps_out=eps_t * MEVFM3_TO_GEOM,
        )
        self._eps_t_mev, self._eps_Q_mev = eps_t, eps_Q

    # ---- accessors ----
    @property
    def segments(self):
        """(inner_quark_table, outer_hadronic_table)."""
        return self._inner, self._outer

    @property
    def raw_segments(self):
        """(inner, outer) with the outer segment on its raw nodes, before
        :func:`refine_table`.  For solvers that interpolate between nodes
        linearly (``reference.scipy_tov``), where added nodes change p(eps)."""
        return self._inner, self._outer_raw

    @property
    def transition(self) -> TransitionInfo:
        return self._transition

    def eos_table(self) -> EOSTable:
        """Combined table (jump as a cs^2->0 cell); for adapter / plotting."""
        no, eo, po = self._had_seg
        nq, eq, pq = self._quark_seg
        n = np.concatenate([no, nq])
        eps = np.concatenate([eo, eq])
        p = np.concatenate([po, pq])
        if self._n_outer is None:
            return build_h_and_cs2(n, eps, p)
        return refine_table(build_h_and_cs2(n, eps, p, xp=np),
                            self._n_outer + self.n_quark)

    def eos_table_smeared(self, width_frac=0.02, n_ramp=60) -> EOSTable:
        """Replace the sharp jump with a finite-width crossover in pressure.

        The transition is spread over dp = width_frac * p_t, with eps rising
        eps_out -> eps_in across it (cs^2 = dp/deps small but finite).  As
        width_frac -> 0 this approaches the sharp transition; the standard adaptive
        solver resolves it, giving an independent value to compare the junction
        solver against.
        """
        no, eo, po = self._had_seg
        nq, eq, pq = self._quark_seg
        pt = float(self.p_trans_mev)
        dp = width_frac * pt
        # ramp at constant-ish small cs^2 between the two eps values
        eps_ramp = np.linspace(self._eps_t_mev, self._eps_Q_mev, n_ramp)
        p_ramp = pt + dp * (eps_ramp - self._eps_t_mev) / (
            self._eps_Q_mev - self._eps_t_mev)
        n_ramp_arr = np.linspace(no[-1], nq[0], n_ramp)
        # quark branch shifted to start at p_t + dp
        pq2 = pq + dp
        n = np.concatenate([no[:-1], n_ramp_arr, nq])
        eps = np.concatenate([eo[:-1], eps_ramp, eq])
        p = np.concatenate([po[:-1], p_ramp, pq2])
        # enforce strictly increasing p for the table builder
        p = np.maximum.accumulate(p) + 1e-12 * np.arange(len(p))
        return build_h_and_cs2(n, eps, p)
