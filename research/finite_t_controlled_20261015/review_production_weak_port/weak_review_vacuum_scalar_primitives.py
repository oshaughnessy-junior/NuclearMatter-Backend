# Generatedselecteddefinitions; environmentexplicitinoracles.py; notupstreamdriver.
@dataclass(frozen=True)
class Channel:
    initial_mass: float
    final_mass: float
    lepton_mass: float
    antineutrino: bool = False
    gv: float = 1.0
    ga: float = 1.2723

    def __post_init__(self):
        vals = (self.initial_mass, self.final_mass, self.lepton_mass, self.gv, self.ga)
        if not all((math.isfinite(x) for x in vals)) or min(vals[:2]) <= 0 or self.lepton_mass < 0:
            raise ValueError('Finite positive nucleon masses, nonnegative lepton mass required')
        if not isinstance(self.antineutrino, bool):
            raise ValueError('Explicit neutrino/antineutrino bool required')

    @property
    def threshold(self):
        a, b, m = (self.initial_mass, self.final_mass, self.lepton_mass)
        return max(0.0, (b + m - a) * (b + m + a) / (2 * a))

def dot(a, b):
    return float(np.dot(METRIC * a, b))
