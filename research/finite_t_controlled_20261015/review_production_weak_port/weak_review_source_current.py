# Generatedselecteddefinitions; environmentexplicitinoracles.py; notupstreamdriver.
@dataclass(frozen=True)
class FormFactors:
    gv: float = 1.0
    ga: float = 1.2723
    xi: float = 3.706
    vector_mass: float = 840.0
    axial_mass: float = 1000.0
    pion_mass: float = 139.57
    weak_magnetism: bool = True
    pseudoscalar: bool = True
    q2_dependence: bool = True

    def __post_init__(self):
        if not all((math.isfinite(x) for x in (self.gv, self.ga, self.xi, self.vector_mass, self.axial_mass, self.pion_mass))) or min(self.vector_mass, self.axial_mass, self.pion_mass) <= 0:
            raise ValueError('Finite couplings and positive pole masses required')
        if not all((isinstance(x, bool) for x in (self.weak_magnetism, self.pseudoscalar, self.q2_dependence))):
            raise ValueError('Explicit bool current switches required')

    def values(self, t, M):
        if not math.isfinite(t) or not math.isfinite(M) or M <= 0:
            raise ValueError('Finite t and positive mass normalization required')
        if t >= min(self.pion_mass ** 2, self.vector_mass ** 2, self.axial_mass ** 2, 4 * M * M):
            raise ValueError('Outside declared sub-pole domain')
        if self.q2_dependence:
            den = (1 - t / (4 * M * M)) * (1 - t / self.vector_mass ** 2) ** 2
            gv = self.gv * (1 - (1 + self.xi) * t / (4 * M * M)) / den
            ga = self.ga / (1 - t / self.axial_mass ** 2) ** 2
            ff = self.xi / den
        else:
            gv, ga, ff = (self.gv, self.ga, self.xi)
        if self.pion_mass ** 2 - t <= 0:
            raise ValueError('Outside declared sub-pion-pole domain')
        gp = 2 * M * M * ga / (self.pion_mass ** 2 - t)
        return (gv, ga, ff if self.weak_magnetism else 0.0, gp if self.pseudoscalar else 0.0)

def guo_invariants(channel, p, ff):
    """Independent published Eq47a-h, vacuum q*=q; no calibration factors."""
    v, a, f, g = ff
    k, n, l, b = (p[x] for x in ('k', 'pa', 'l', 'pb'))
    q = b - n
    mn, mp = (channel.initial_mass, channel.final_mass)
    M = (mn + mp) / 2
    kn, kb, ln, lb, kl, nb = (V.dot(x, y) for x, y in [(k, n), (k, b), (l, n), (l, b), (k, l), (n, b)])
    kq, lq, nq, bq, t = (V.dot(x, y) for x, y in [(k, q), (l, q), (n, q), (b, q), (q, q)])
    pref = 16 * V.GF ** 2 * V.VUD ** 2
    s = -1 if channel.antineutrino else 1
    return dict(VV=pref * v * v * (kn * lb + kb * ln - mn * mp * kl), VA=s * 2 * pref * v * a * (kn * lb - kb * ln), AA=pref * a * a * (kn * lb + kb * ln + mn * mp * kl), VF=pref * v * f / (2 * M) * ((kn * mp - kb * mn) * lq + (ln * mp - lb * mn) * kq + (nq * mp - bq * mn) * kl), AF=s * pref * a * f / M * ((kn * mp + kb * mn) * lq - (ln * mp + lb * mn) * kq), FF=pref * f * f / (8 * M * M) * (2 * (ln * bq + lb * nq) * kq + 2 * (kn * bq + kb * nq) * lq - 2 * kq * lq * nb + t * (kl * nb - 2 * kn * lb - 2 * kb * ln) - mn * mp * (kl * t + 2 * kq * lq)), AP=pref * a * g / M * (kq * (ln * mp - lb * mn) + lq * (kn * mp - kb * mn) + kl * (bq * mn - nq * mp)), PP=pref * g * g / (2 * M * M) * (2 * kq * lq - t * kl) * (nb - mn * mp))
